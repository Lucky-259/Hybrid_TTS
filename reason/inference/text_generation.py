from typing import List, Optional
import requests
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from openai import OpenAI


@dataclass
class ConcatedLMGenResult:
    text: List[str]
    prompt_tokens: List[int]
    num_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]
    finish_reason: List[str]

    # post init compute number of completion_tokens
    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)


def _generate_vllm(
    query_str,
    model_name,
    n,
    max_new_tokens,
    stop_token_ids,
    stop_str,
    include_stop_str_in_output
) -> ConcatedLMGenResult:
    

    gen_params_dict = {
        "n": n,
        "max_tokens": max_new_tokens,
        "stop": stop_str,
        "logprobs": 0,
        "seed": 42,
        "extra_body":{
            "include_stop_str_in_output" : include_stop_str_in_output,
            "skip_special_tokens": False
        }
    }
    
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8012/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    results = client.completions.create(model=model_name,
                                        prompt=query_str,
                                        **gen_params_dict)
    outputs =[]
    cum_logps = []
    finish_reason = []
    output_token_lens = []
    avg_len_logps = []

    for i in range(n):
        outputs.append(results.choices[i]) 
        cum_logp = sum(results.choices[i].logprobs.token_logprobs)
        cum_logps.append(cum_logp)
        finish_reason.append(results.choices[i].finish_reason) 
        output_token_len=len(results.choices[i].logprobs.tokens)
        output_token_lens.append(output_token_len)
        avg_len_logps.append(cum_logp / max(1, output_token_len))

     
    prompt = results.usage.prompt_tokens

    return ConcatedLMGenResult(
        text=[i.text for i in outputs],
        prompt_tokens=[prompt],
        num_tokens=output_token_lens,
        cumulative_logprob=cum_logps,
        logp_avg_by_len=avg_len_logps,
        finish_reason=finish_reason,
    )
