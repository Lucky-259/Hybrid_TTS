from dataclasses import dataclass
from datetime import datetime
import importlib
from multiprocessing import Pool
from typing import Any, Callable, Dict, Optional, List, Union
import datetime
import numpy as np
import ray
from envs import get_default_query_str_builder, get_env_datasets
from reason.inference.lm_call import LanguageModelCallingFunction
from reason.inference.rm_call import RewardModelCallingFunction
from reason.reranking.vote_utils import (
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_VOTE,
    PRM_LAST_MAX,
    AGG_FN_MAP,
)
from envs.base_env import INVALID_ANS

class Task:
    def __init__(self, task_name: str, is_few_shot: bool = False,save_dir:str = ''):
        self.task_name = task_name
        self.save_dir=save_dir
        task_module = importlib.import_module(f"envs.{task_name}")
        if task_name == "MATH"  or task_name == "critic_MATH":            
            self.extract_answer = task_module.extract_answer
            self.extract_groundtruth = task_module.extract_groundtruth
            self.judge_correct = task_module.judge_correct   
        else:
            raise NotImplementedError(f"Task {task_name} is not supported")

        self._is_few_shot = is_few_shot
        self.env_fn = task_module.Env

    def prompt_fn(self, problem_input: str):
        return get_default_query_str_builder(self.task_name)(
            problem_input, is_few_shot=self._is_few_shot
        )

    @property
    def test_ds(self):
        return get_env_datasets(self.task_name,self.save_dir)[1]


CHOSEN_AGGR_METHODS = [
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_MAX,
    PRM_LAST_VOTE,
]


def judge_ans(
    problem_str: str,
    extracted_groundtruth: str,
    output_list: List[str],
    v_list: List[float],
    aggration_mode: str,
    extract_answer_fn,
    judge_correct_fn,
    normalize=False,
):
    ans_list = [extract_answer_fn(txt) for txt in output_list]
    print("------------------aggration_mode,ans_list------------------\n\n")
    print(aggration_mode,ans_list)
    print("-----------------------------------------------------------\n\n")
    valid_ans_list, valid_v_list = [], []
    for i, ans in enumerate(ans_list):
        if ans != INVALID_ANS:
            valid_ans_list.append(ans)
            valid_v_list.append(v_list[i])
    if len(valid_ans_list) == 0:
        return 0

    if "orm" in aggration_mode and normalize:
        valid_v_list = np.array(valid_v_list)
        valid_v_list -= valid_v_list.min()
        valid_v_list /= valid_v_list.max() + 1e-3
        valid_v_list = valid_v_list.tolist()
    aggregated_ans = AGG_FN_MAP[aggration_mode](valid_ans_list, valid_v_list)
    print("------------------aggration_mode,extracted_groundtruth,aggregated_ans,valid_ans_list,valid_v_list,judge_correct------------------\n\n")
    print(aggration_mode, extracted_groundtruth,aggregated_ans,valid_ans_list, valid_v_list,judge_correct_fn(problem_str, extracted_groundtruth, aggregated_ans))
    print("---------------------------------------------------------------------------------------------------------------------------------\n\n")
    return (
        1 if judge_correct_fn(problem_str, extracted_groundtruth, aggregated_ans) else 0
    )

def get_ans(
    problem_str: str,
    extracted_groundtruth: str,
    output_list: List[str],
    v_list: List[float],
    aggration_mode: str,
    extract_answer_fn,
    judge_correct_fn,
    normalize=False,
):
    ans_list = [txt for txt in output_list]
    
    valid_ans_list, valid_v_list = [], []
    for i, ans in enumerate(ans_list):
        if ans != INVALID_ANS:
            valid_ans_list.append(ans)
            valid_v_list.append(v_list[i])
    if len(valid_ans_list) == 0:
        return 0

    if "orm" in aggration_mode and normalize:
        # score_normalization: this is only necessary for [-1, 1] values
        valid_v_list = np.array(valid_v_list)
        valid_v_list -= valid_v_list.min()
        valid_v_list /= valid_v_list.max() + 1e-3
        valid_v_list = valid_v_list.tolist()
    aggregated_ans = AGG_FN_MAP[aggration_mode](valid_ans_list, valid_v_list)
    
    return aggregated_ans

@dataclass
class SolutionOutput:
    solutions: List[str]
    # Define the completion tokens for each solution
    #  For best_of_n, it's a list of int, indicate how many tokens in each
    #      generation
    #  for beam search, it's a list of zeros, except the last element indicates total tokens
    #  for mcts, it's a list of int, indicate how many tokens comsumed between two paths
    completion_tokens: List[int]
    path_tokens: List[int]
    metadata: Optional[Dict] = None      # other information


@dataclass
class TreeSearchSolutionOutput(SolutionOutput):
    tree_completion_tokens: List[int]=None

class MathEvaluator:

    def __init__(
        self,
        task: Union[str, Task],
        save_dir,
        lm_call: LanguageModelCallingFunction,
        rm_call: RewardModelCallingFunction,
    ):
        if isinstance(task, str):
            self._task = Task(task_name=task,save_dir=save_dir)
        else:
            assert isinstance(task, Task)
            self._task = task
        self.lm_call = lm_call
        self.rm_call = rm_call

    def evaluate_problem(
        self, problem_inst: Dict[str, str], solver_fn: Callable, save_dir
    ) -> (str, str, str, Dict[str, str]):
        import jsonlines
        if save_dir is not None:
            record_writer = jsonlines.open(save_dir / f"record_backup.jsonl", mode="a")
        else:
            record_writer = None
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'------------------{problem_inst["question"][0:14]} thread START:{current_time}------------------\n')
        solution: SolutionOutput = solver_fn(problem_inst, self.lm_call, self.rm_call)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'------------------{problem_inst["question"][0:14]} thread END:{current_time}------------------\n')
        result, output, select_res, final_sel = self.analyze_output(problem_inst, solution.solutions, solution.path_tokens)
        total_completion_token = 0
        for i, o in enumerate(output):
            o["completion_tokens"] = solution.completion_tokens[i]
            if isinstance(solution, TreeSearchSolutionOutput):
                o["tree_completion_tokens"] = solution.tree_completion_tokens[i]
            total_completion_token += solution.completion_tokens[i]
        result["total_completion_tokens"] = total_completion_token

        processed_metadata = self.process_meta_data(solution.metadata)

        
        if record_writer:
            obj = {
                "question": problem_inst["question"],
                "groundtruth": problem_inst["answer"],
                "result": result,
                "output": output,
                "select_res": select_res,
                "final_select_answer": final_sel,
                **processed_metadata

            }
            record_writer.write(obj)
        return problem_inst, result, output, select_res, final_sel, processed_metadata

    def analyze_output(self, problem_inst: Dict[str, str], gen_answers: List[str], path_tokens):
        extracted_groundtruth = self._task.extract_groundtruth(problem_inst["answer"])

        if len(gen_answers) > 1:
            input_list = [(problem_inst["question"], txt) for txt in gen_answers]

            value_list = self.rm_call(input_list, lm_step_tag=self.lm_call.lm_step_tag)
        else:
            value_list = [[0]]
        output_list = [
            {"path_idx": i, "text": txt, "value": v, "path_tokens": p}
            for i, (txt, v, p) in enumerate(zip(gen_answers, value_list, path_tokens))
        ]
        txt_dict={}
        for i in output_list:
            txt_dict[i["text"]] = i["path_tokens"]
        res = {
            agg_method: judge_ans(
                problem_inst["question"],
                extracted_groundtruth,
                gen_answers,
                value_list,
                agg_method,
                self._task.extract_answer,
                self._task.judge_correct,
            )
            for agg_method in (
                CHOSEN_AGGR_METHODS if len(gen_answers) > 1 else [MAJORITY_VOTE]
            )
        }
        select_res = {
            agg_method: get_ans(
                problem_inst["question"],
                extracted_groundtruth,
                gen_answers,
                value_list, 
                agg_method,
                self._task.extract_answer,
                self._task.judge_correct,
            )
            for agg_method in (
                CHOSEN_AGGR_METHODS if len(gen_answers) > 1 else [MAJORITY_VOTE]
            )
        }
        for agg_method in (CHOSEN_AGGR_METHODS if len(gen_answers) > 1 else [MAJORITY_VOTE]):
            agg_answer = select_res[agg_method]
            agg_tokens = txt_dict[agg_answer]
            select_res[agg_method] = (agg_answer, agg_tokens)
        
        for key,val in res.items():
            final_sel = (select_res[key] ,val)
            if val == 1:
                break

        return res, output_list, select_res, final_sel

    def process_meta_data(self, metadata_dict=None):
        if metadata_dict is None:
            return {}

        return metadata_dict


@ray.remote
class RemoteMathEvaluator(MathEvaluator):
    def __init__(
        self,
        task: str,
        lm_call: LanguageModelCallingFunction,
        rm_call: RewardModelCallingFunction,
    ):
        print(f"Initializing RemoteMathEvaluator with task={task}")
        super().__init__(task, lm_call, rm_call)
        print("RemoteMathEvaluator initialized")
