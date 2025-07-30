from typing import Union, List
import requests
import torch.nn.functional as F

# Qwen2.5-PRM online
def _value_inference(
    model,
    tokenizer,
    input_str: Union[List[str], str],
):
    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) 
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] 
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] 
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
    
    def post_http_request(prompt: dict, api_url: str) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        response = requests.post(api_url, headers=headers, json=prompt)
        return response

    api_url = f"http://localhost:8011/pooling"
    step_reward = []
    for input in input_str:
        question_answer_pair = input.split("\n\n\n\n")
        data = {
        "system": "Please reason step by step, and put your final answer within \\boxed{}.",
        "query": question_answer_pair[0],
        "response": [
            answer.strip() for answer in question_answer_pair[1].split("\n\n") if answer != ""
            ]
        }
        
        messages = [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
            {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
        ]
        conversation_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )

        input_ids = tokenizer.encode(
            conversation_str, 
            return_tensors="pt", 
        )
        prompt = {
            "model": model,
            "messages": messages
        }
        pooling_response = post_http_request(prompt=prompt, api_url=api_url)

        outputjson = pooling_response.json()
        outputdata = outputjson['data'][0]['data']
        step=[]
        for i in outputdata:
            step.append(i[1])
        step_reward.append(step)

    return step_reward

# # math-shepherd-mistral-7b-prm offline
# def _value_inference(
#     model,
#     tokenizer,
#     input_str: Union[List[str], str],
# ):
#     good_token = '+'
#     bad_token = '-'
#     step_tag = 'ки'

#     candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
#     step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902

#     step_reward = []
#     for input in input_str:
#         question_answer_pair = input.split("\n\n\n\n")
#         question = question_answer_pair[0]
#         answer = question_answer_pair[1]
#         input_for_prm = f"{question} {answer}"
#         input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to(model.device)

#         with torch.no_grad():
#             logits = model(input_id).logits[:,:,candidate_tokens]
#             scores = logits.softmax(dim=-1)[:,:,0]
#             step_reward.append(scores[input_id == step_tag_id].tolist())

#         # print(step_reward)  # [[1.0, 0.1904296875, 0.9765625, 1.0]]

#     return step_reward