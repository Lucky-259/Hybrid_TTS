from typing import List, Dict, Tuple
import os
import numpy as np
import json
import re
from envs.MATH.env import CoTEnv
from envs.base_env import NoLegalActionException, ResetException
from tqdm import tqdm
from envs.MATH.env import extract_answer, extract_groundtruth, judge_correct
from reason.inference.lm_call import LMCallingConfig

from .Qwen_step_prompt import (
    COT_TASK_DESC,
    CRITIQUE_TASK_DESC,
    REWRITE_TASK_DESC,
    COT_FORMAT_STR,
    CRITIQUE_FORMAT_STR,
    REWRITE_FORMAT_STR,
    SEP,
)

from distributed.utils import print_with_rank
from loguru import logger

from pathlib import Path
import time
# Get the file path of the current script
CURRENT_DIR = Path(__file__).parent


ANS_RE = None
STOP_STR = None


def read_txt(file_path):
    assert str(file_path).endswith(".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def read_json(file_path):
    assert str(file_path).endswith(".json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class Env(CoTEnv):
    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        task_desc_str: str = None,
        cot_example_str: str = None,
        problem_format_str: str = None,
        cot_task_desc_str: str = COT_TASK_DESC,
        critique_task_desc_str: str = CRITIQUE_TASK_DESC,
        rewrite_task_desc_str: str = REWRITE_TASK_DESC,
        cot_format_str: str = COT_FORMAT_STR,
        critique_format_str: str = CRITIQUE_FORMAT_STR,
        rewrite_format_str: str = REWRITE_FORMAT_STR,
        reset=False,
    ):
        """
        three types of thinking in reasoning:
        1. i.i.d sampling: generative independent response
        2. conditional sampling (counterfactual): what if the response is wrong?
        3. reflective sampling
        Args:
            config:
            math_problems:
            llm_gen_fn:
            task_desc_str:
            cot_example_str:
            problem_format_str:
            reset:
        """
        super().__init__(
            config,
            math_problems,
            llm_gen_fn,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset=False,
        )

        self.current_node_type = None
        self.num_first_answer = config["num_sample"]
        self.num_refine = config["num_refine"]
        self.prm_threshold = config["prm_threshold"]
        self.refine_cut_num = config["refine_cut_num"]
        self.prm_cut = config["prm_gap"]
        self.rewritefrom = config["rewrite_from"] 

        self.num_reviews = 3
        self.num_rewrite = 3
        self.max_new_tokens_answer = 1024
        self.max_new_tokens_review = 1024
        self.max_depth = 5

        self.print_log = True
        self.total_api_call_completion = 0
        self.total_tree_completion = 0

        self.sep = "\n\n"
        self._init_query = None
        self._next_state_terminated = None

        if 'Qwen' in llm_gen_fn.modelname():
            from .Qwen_step_prompt import (
                COT_TASK_DESC,
                CRITIQUE_TASK_DESC,
                REWRITE_TASK_DESC,
                COT_FORMAT_STR,
                CRITIQUE_FORMAT_STR,
                REWRITE_FORMAT_STR,
                SEP,
            )
            self.eos = "<|im_end|>"
        elif 'Llama' in llm_gen_fn.modelname():
            from .Llama_step_prompt import (
                COT_TASK_DESC,
                CRITIQUE_TASK_DESC,
                REWRITE_TASK_DESC,
                COT_FORMAT_STR,
                CRITIQUE_FORMAT_STR,
                REWRITE_FORMAT_STR,
                SEP,
            )
            self.eos = '<|eot_id|>'
        elif 'gemma' in llm_gen_fn.modelname():
            from .Gemma_step_prompt import (
                COT_TASK_DESC,
                CRITIQUE_TASK_DESC,
                REWRITE_TASK_DESC,
                COT_FORMAT_STR,
                CRITIQUE_FORMAT_STR,
                REWRITE_FORMAT_STR,
                SEP,
            )
            self.eos = '<end_of_turn>'
        self.cot_task_desc = COT_TASK_DESC
        self.critique_task_desc = CRITIQUE_TASK_DESC
        self.rewrite_task_desc = REWRITE_TASK_DESC

        self.cot_format_str = COT_FORMAT_STR
        self.critique_format_str = CRITIQUE_FORMAT_STR
        self.rewrite_format_str = REWRITE_FORMAT_STR

        if reset:
            self.reset(update_legal_action=True)
        # self.rewrite_prompt_template = None

    def check_attribute(self):
        assert hasattr(self, "cot_task_desc")
        assert hasattr(self, "critique_task_desc")
        assert hasattr(self, "rewrite_task_desc")
        assert hasattr(self, "cot_format_str")
        assert hasattr(self, "critique_format_str")
        assert hasattr(self, "rewrite_format_str")

    @property
    def stop_str(self):
        return STOP_STR

    @property
    def answer(self):
        return ""

    @property
    def full_answer(self):
        return self.action_history[-1]

    def post_process_act(self, action: str):
        if not action.endswith(self.sep):
            action = action.strip() + self.sep

        return action

    def set_problem(self, idx):
        self.math_problem = self.math_problems[idx]

    def get_state(self):
        # if no solution has been generated yet, generate the initial query
        if len(self.action_history) == 0:
            ret = self.cot_task_desc + self.cot_format_str.format(
                question=self.question
            )
        else:
            ret = (
                self.cot_task_desc
                + self.cot_format_str.format(question=self.question)
                + self.action_history[-1]
            )
        return ret

    def reset(self, update_legal_action=True):
        """
        reset the environment, and generate the first solution to the question
        Args:
            update_legal_action:
        Returns:
        """
        assert update_legal_action, print("Need to first update legal action")
        self.set_problem(idx=0)
        self.action_history = []
        self.review_history = []

        if update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = (
                        self.generate_first_response()
                    )
                    break
                except NoLegalActionException as e:
                    if cnt == 3:
                        raise ResetException
        info = {"api_completion_token": api_completion_token}
        return None, info

    def generate_first_response(self) -> (List[Dict], int):
        first_cot_prompt = self.cot_task_desc + self.cot_format_str.format(
           question=self.math_problem["question"]
        )
        result = self.llm_gen_fn(
            input_str=first_cot_prompt,
            config=LMCallingConfig(
                n=self.num_first_answer,
                stop_str=self.sep,
                include_stop_str_in_output=True,
                max_new_tokens=self.max_new_tokens_answer,
                
            ),
        )
        
        texts = result.text
        logps_avg_by_len = result.logp_avg_by_len
        token_len = result.num_tokens

        _legal_actions = [
            {
                "action": action,
                "prob": prob,
                "num_token": n_token,
                "finish_reason": finish_reason,
                "from_review": "",
            }
            for action, prob, n_token, finish_reason in zip(
                texts, logps_avg_by_len, token_len, result.finish_reason
            )
        ]
        self._next_state_terminated = dict(zip(texts, [False] * len(texts)))
        return _legal_actions, result.completion_tokens

    def step(self, action, prm, reward_fn, update_legal_action=True):
        """
        Args:
            action: the chosen action, which is the refined solution in this case, need to record this
            update_legal_action:
        Returns:
        """

        self.action_history.append(action)  # recording all the select full answer
        rewrite_text = action
        refine_tokens = 0
        reward = self.get_reward() # 0
        step_rewrite_list, step_prm_list = [action], [prm]
        if prm > self.prm_threshold:
            terminated, truncated, info = (
                self.get_done_and_info(action)
            ) # terminated or truncated when reach maximim depth
        else:
            terminated, truncated, info = (
                self.get_done_and_info(" ")
            )

        # update legal actions
        if not (terminated or truncated) and update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    terminated, truncated, info,self._legal_actions, api_completion_token, rewrite_text, refine_tokens, step_rewrite_list, step_prm_list = (
                        self.update_legal_actions(prm,reward_fn)
                    )
                    info["api_completion_token"] = api_completion_token
                    break
                except NoLegalActionException as e:
                    if cnt == 3:
                        terminated = True
                        reward = 0
                        self._legal_actions = None
                        info["winner"] = 2
                        info["api_completion_token"] = 0
                    else:
                        pass
        else:
            self._legal_actions = None
            if info["winner"] == 1:
                reward = 1.0
            info["api_completion_token"] = 0
        
        return step_rewrite_list, step_prm_list, refine_tokens, rewrite_text, reward, terminated, truncated, info

    def update_legal_actions(self, prm, reward_fn):
        """
        Given the current state (the completed solution to Q), a critic LLM generate review and
        a proposer LLM rewrite the answer, which is the new updated legal action
        Returns:

        """
        # retrive current answer
        assert len(self.action_history) > 0
        current_action = self.action_history[-1]
        rewrite_text = current_action
        P = self.prm_threshold
        refine_num = self.num_refine
        refine_i = 0
        review_texts=""
        refine_tokens=0
        from_review_text = []
        for _ in range(self.num_first_answer):
            from_review_text.append(review_texts)
        n_prm = prm

        step_rewrite_list = [current_action]
        step_prm_list = [prm]

        refine_cut = 0
        refine_cut_num = self.refine_cut_num
        prm_cut =self.prm_cut

        print(f"'IsRefine': {prm <= P and refine_i < refine_num}")
        while prm <= P and refine_i < refine_num:
        # review
            if self.rewritefrom == 'critic':
                review_prompt = self.critique_task_desc + self.critique_format_str.format(
                    question=self.math_problem["question"]
                    +self.get_lastaction_str(), answer=rewrite_text
                )

                result = self.llm_gen_fn(
                    input_str=review_prompt,
                    config=LMCallingConfig(
                        n=1,
                        stop_str=None,
                        include_stop_str_in_output=False,
                        max_new_tokens=self.max_new_tokens_answer,
                    ),
                )
                refine_tokens += result.completion_tokens
                review_texts = result.text[0]
                rewrite_prompt = self.rewrite_task_desc + self.rewrite_format_str.format(
                    question=self.math_problem["question"] ,
                    answer=rewrite_text,
                    review=review_texts,
                )+self.get_lastaction_str()
            elif self.rewritefrom == 'prm':
                
                rewrite_prompt = self.prm_task_desc.format(prm=prm) + self.prm_format_str.format(
                    question=self.math_problem["question"] ,
                    answer=rewrite_text,
                )+self.get_lastaction_str()
            else:
                raise Exception('rewrite from module error!')

            result = self.llm_gen_fn(
                input_str=rewrite_prompt,
                config=LMCallingConfig(
                    n=1,#self.num_rewrite = 3
                    stop_str=self.sep,#"\n\n",
                    include_stop_str_in_output=True,
                    max_new_tokens=self.max_new_tokens_answer,
                ),
            )
            refine_tokens += result.completion_tokens
            
            prms = reward_fn(
                [
                    (
                        self.math_problem["question"]+self.get_lastaction_str(),
                        result.text[0],
                    )
                    
                ]
            )

            n_prm = prms[0][0]
            if n_prm>prm:
                from_review_text = []
                for _ in range(self.num_first_answer):
                    from_review_text.append(review_texts)
                rewrite_text = result.text[0]
                prm = n_prm
            
            if n_prm-prm< prm_cut:
                refine_cut+=1
            else:
                refine_cut=0
            if refine_cut == refine_cut_num+1:
                refine_i = refine_num

            step_rewrite_list.append(rewrite_text)
            step_prm_list.append(prm)

            refine_i += 1
        self.action_history[-1] = rewrite_text

        terminated, truncated, info = (
                self.get_done_and_info(rewrite_text)
            )
        _legal_actions = []

        total_completion = refine_tokens
        if not terminated:    
            cot_prompt = self.cot_task_desc + self.cot_format_str.format(
                question=self.math_problem["question"]
            )+self.get_lastaction_str()+rewrite_text+"\n"
            a_try=1
            max_a_try=6
            try:
                result = self.llm_gen_fn(
                    input_str=cot_prompt,
                    config=LMCallingConfig(
                        n=self.num_first_answer,
                        stop_str=self.sep,
                        include_stop_str_in_output=True,
                        max_new_tokens=self.max_new_tokens_answer,
                    ),
                )
            except Exception as e:
                while True:
                    a_try+=1
                    time.sleep(2)
                    try:
                        result = self.llm_gen_fn(
                            input_str=cot_prompt,
                            config=LMCallingConfig(
                                n=self.num_first_answer,
                                stop_str=self.sep,
                                include_stop_str_in_output=True,
                                max_new_tokens=self.max_new_tokens_answer,
                            ),
                        )
                        break
                    except:
                        if a_try<max_a_try:
                            continue
                        else:
                            raise e
                print("CONTEXT LIMITATION:{cot_prompt}")
                raise e
            
            new_action_text = []
            
            new_prob_list = []
            tokens_num_list = []
            final_reason_list = []
            
            new_action_text += result.text 
            new_prob_list += result.logp_avg_by_len
            tokens_num_list += result.num_tokens
            final_reason_list += result.finish_reason
            total_completion += result.completion_tokens

            _legal_actions = [
                {
                    "action": action,
                    "prob": prob,
                    "num_token": n_token,
                    "finish_reason": finish_reason,
                    "from_review": f_review,
                }
                for action, prob, n_token, finish_reason, f_review in zip(
                    new_action_text,
                    new_prob_list,
                    tokens_num_list,
                    final_reason_list,
                    from_review_text,
                )
            ]

            self._next_state_terminated = dict(
                zip(new_action_text, [False] * len(new_action_text))
            )

        return terminated, truncated, info,  _legal_actions, total_completion, rewrite_text, refine_tokens, step_rewrite_list, step_prm_list

    def get_done_and_info(self,action):
        info = {"winner": 0}
        last_action = action
        truncated = terminated = self.eos  in last_action or len(self.action_history) >= 50  
        if len(self.action_history)>15 and self.action_history[-1] == self.action_history[-2] and self.action_history[-2] == self.action_history[-3]:
            truncated = terminated = True
        if len(self.action_history)>15 and self.action_history[-1] == self.action_history[-3] and self.action_history[-3] == self.action_history[-5]:
            truncated = terminated = True    
        action_list = last_action.split(" ")
        if len(action_list)>35 and len(set(action_list))<8:
            truncated = terminated = True
        num=0
        action_list_2 = last_action.split("\n")
        for i in action_list_2:
            if 'step' in i.lower():
                num+=1
        if num>8:
            truncated = terminated = True 
        action_other_list = last_action.split("=")
        if len(action_other_list)>50 and len(set(action_other_list))<8:
            truncated = terminated = True
        if terminated or truncated:
            if self._is_correct(self.action_history[-1]):
                info["winner"] = 1
            else:
                info["winner"] = 2
            return terminated, truncated, info
        return terminated, truncated, info

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        return judge_correct(
            self.math_problem["question"], self.math_problem["answer"], extracted_answer
        )
    def get_lastaction_str(self):
        s = ""
        for i in self.action_history[:len(self.action_history)-1]:
            s = s +"\n"+  i
        return s
    def get_allaction_str(self):
        s = ""
        for i in self.action_history[:len(self.action_history)]:
            s = s +"\n"+  i 
        return s
    def get_reward(self):
        """To implement based on learned reward model"""
        return 0
