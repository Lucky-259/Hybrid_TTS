from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from config_utils import str2bool
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.inference.rm_call import (
    RMRemoteCaller,
    DummyRewardModelCaller,
    RewardModelBaseConfig,
    RemoteRewardModelConfig,
)
from reason.evaluation.evaluator import SolutionOutput, Task, RemoteMathEvaluator, MathEvaluator
import torch
from functools import partial
import json
import jsonlines
import time
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import os
import random
from multiprocessing import Pool
import tree
from ray.util.actor_pool import ActorPool
from reason.evaluation.methods import *
import ray

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--LM", type=str, required=True)
    parser.add_argument("--RM", type=str, default="dummy")
    # task config
    parser.add_argument("--task_name", type=str, default="gsm8k")
    parser.add_argument("--test", type=str2bool, default=True)
    parser.add_argument("--is_few_shot", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    # method config
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--num_sequence", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    # Tree construction config
    parser.add_argument("--tree_max_depth", type=int, default=None)
    parser.add_argument("--tree_max_width", type=int, default=None)
    # ckpg config
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)
    # parallel config
    parser.add_argument("--local", action="store_true", default=False)

    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--num_refine", type=int, default=1)
    parser.add_argument("--prm_threshold", type=float, default=0.5)
    parser.add_argument("--refine_cut_num", type=int, default=5)
    parser.add_argument("--prm_gap", type=float, default=0.1)
    parser.add_argument("--rewrite_from", type=str, default='critic')

    config = parser.parse_args()
    setup_seed(config.seed)

    # TODO(ziyu): move into some configuration file
    if "math-shepherd" in config.RM.lower():
        prm_step_tag = "ки\n"
    else:
        # assume qwen
        prm_step_tag = "\n\n"
    prm_format_str = "{question}\n\n\n\n{answer}"

    if "qwen" in config.LM.lower():
        lm_step_tag = "\n\n"
    else:
        lm_step_tag = "ки\n"

    llm_gen_fn = VLLMRemoteCaller(
        config.LM, lm_step_tag=lm_step_tag
    )
    if config.RM == "dummy":
        rm_config = RewardModelBaseConfig(
            step_tag=prm_step_tag, format_str=prm_format_str
        )
        rm_call = DummyRewardModelCaller(rm_config)
    else:
        rm_config = RemoteRewardModelConfig(
            step_tag=prm_step_tag,
            format_str=prm_format_str,
            model_name=config.RM,
            # controller_addr=config.controller_addr,
        )
        rm_call = RMRemoteCaller(rm_config)

    task = Task(task_name=config.task_name, is_few_shot=config.is_few_shot,save_dir=config.save_dir)

    def parallel_evaluate_test_dataset(
        method_name: str, solver_fn: Callable, save_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"------------------BEGIN:{current_time}------------------\n\n")
        if save_dir is not None:
            record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="w")
        else:
            record_writer = None

        test_ds = task.test_ds

        results = []
        selt = []
        f_selt = []
        if config.resume_dir is not None:
            answered_questions = set()
            with jsonlines.open(
                Path(config.resume_dir) / "record.jsonl", "r"
            ) as reader:
                cnt = 0
                for obj in reader:
                    results.append(obj["result"])
                    answered_questions.add(obj["question"])
                    if record_writer is not None:
                        record_writer.write(obj)
                        cnt += 1
            print(f"Resumed {cnt} questions from {config.resume_dir}")
            total_cnt = len(test_ds)
            test_ds = [
                problem_inst
                for problem_inst in test_ds
                if problem_inst["question"] not in answered_questions
            ]
            new_cnt = len(test_ds)
            print(
                f"After resuming, there are {new_cnt}/{total_cnt} new questions to answer."
            )
        myevaluator = MathEvaluator(config.task_name,save_dir, llm_gen_fn, rm_call)

        res_q=[]
        
        # more thread
        from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
        with ThreadPoolExecutor(max_workers=32) as executor:
            for idx,test_ds_i in enumerate(test_ds):
                res_q.append(executor.submit(myevaluator.evaluate_problem, test_ds_i, solver_fn, save_dir))
            wait(res_q, return_when=ALL_COMPLETED)
        final_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"------------------END:{final_time}------------------\n\n")
            
        
        # #results.append(res_q.result)
        for i in range(len(res_q)):
            res_q[i] = res_q[i].result()
        # more thread
        '''
        # # # one thread
        for idx, test_ds_i in enumerate(test_ds):
            res_q.append(myevaluator.evaluate_problem(test_ds_i, solver_fn))
        '''
        # # # one thread
        #res_q.append(myevaluator.evaluate_problem(test_ds[20], solver_fn, save_dir))
        for i, (problem_inst, result, output, select_res, final_sel, metadata_dict) in enumerate(
            tqdm(res_q, total=len(res_q))
        ):
            
            results.append(result)
            select_res_dic = {}
            for key,value in select_res.items():
                key+="_tokens"
                select_res_dic[key] = value[1]
            selt.append(select_res_dic)
            step_answer ={}
            try:
                step_info = metadata_dict["refine_info_list"][0]
            
                for i in range(len(step_info)):
                    if i%2==0:
                        step_answer[f"step{i//2+1}"]=[step_info[i],step_info[i+1]]
            except:
                step_answer ={}
            
            f_selt.append({"final_select_answer_tokens":final_sel[0][1],"final_select_answer":final_sel[1]})
            if record_writer:
                obj = {
                    "question": problem_inst["question"],
                    "groundtruth": problem_inst["answer"],
                    "result": result,
                    "output": output,
                    "select_res": select_res,
                    "final_select_answer": final_sel,
                    "step_answer": step_answer,
                    **metadata_dict
                }
                record_writer.write(obj)
        avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *results),)
        selt_avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *selt),)
        f_selt_avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *f_selt),)
        if record_writer:
            json.dump(avg_res, open(save_dir / "avg_result.json", "w"))
            json.dump(selt_avg_res, open(save_dir / "avg_result.json", "w"))
            json.dump(f_selt_avg_res, open(save_dir / "avg_result.json", "w"))
        print("Method: {}. Average result: {}, {}, {}".format(method_name, avg_res, selt_avg_res, f_selt_avg_res))
        return results

    solver_fns = {"cot": cot, "best_of_n": best_of_n}

    cfg_dict_record = dict()
    gen_config = LMCallingConfig(
        n=config.num_sequence,
        max_new_tokens=config.max_new_tokens,
        num_sample = config.num_sample,
        num_refine = config.num_refine,
        prm_threshold = config.prm_threshold,
        refine_cut_num=config.refine_cut_num,
        prm_gap=config.prm_gap,
        rewrite_from = config.rewrite_from 
    )
    cfg_dict_record["gen_config"] = gen_config.__dict__

    if config.method == "cot":
        method_config = CoTConfig(config.task_name)
        solver_fn = partial(cot, method_config, gen_config)
    elif config.method == "best_of_n":
        method_config = BestOfNConfig(
            config.task_name, num_sequence=config.num_sequence
        )
        solver_fn = partial(best_of_n, method_config, gen_config)
    elif config.method == "beam_search":
        method_config = BeamSearchConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            beam_size=config.num_sequence,
        )
        solver_fn = partial(beam_search, method_config, gen_config)
    elif config.method == "vanila_mcts":
        method_config = VanilaMCTSConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            select_by_prior=False,
            num_path=config.num_sequence,
        )
        solver_fn = partial(vanila_mcts, method_config, gen_config)
    elif config.method == "critic_mcts":
        method_config = VanilaMCTSConfig(
            save_dir=config.save_dir,
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            select_by_prior=False,
            num_path=config.num_sequence,
        )
        solver_fn = partial(critic_mcts, method_config, gen_config)
        print("Step-level TTS START")

    else:
        raise ValueError(f"Unknown method: {config.method}")
    cfg_dict_record["method"] = config.method
    cfg_dict_record["method_config"] = method_config.__dict__

    if config.save_dir is not None:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(config.save_dir) / datetime_str
        save_dir.mkdir(parents=True)
        record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="w")
        cfg_dict_record["LM"] = config.LM
        cfg_dict_record["RM"] = config.RM
        json.dump(cfg_dict_record, open(save_dir / "config.json", "w"))
    else:
        save_dir = None
    parallel_evaluate_test_dataset(config.method, solver_fn, save_dir)
