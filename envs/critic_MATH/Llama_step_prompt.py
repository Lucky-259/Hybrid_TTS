# Common
COT_TASK_DESC = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nPlease reason step by step, and put your final answer within \\\\boxed{{}}.<|eot_id|>"""

# GPQA
# COT_TASK_DESC = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nPlease reason step by step, and put your final answer within \\\\boxed{{$LETTER}}(without quotes) where LETTER is one of ABCD.<|eot_id|>"""

# step-level
CRITIQUE_TASK_DESC = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nThere is a weak reasoning step in the solution, please provide a strict reflection to correct only this one step with less than 150 tokens. Don't output the complete solution.<|eot_id|>" 

# Common
REWRITE_TASK_DESC = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nPlease refine the weak answer according to your Reflection. Not allowed to use code to solve the question. Please reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>"

# GPQA
# REWRITE_TASK_DESC = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nPlease refine the weak answer according to your Reflection. Not allowed to use code to solve the question. Please reason step by step, and put your final answer within \\boxed{{$LETTER}}(without quotes) where LETTER is one of ABCD.<|eot_id|>"


COT_FORMAT_STR = (
    """<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"""
)

CRITIQUE_FORMAT_STR = """<|start_header_id|>user<|end_header_id|>\nQuestion: {question}\nThe weak reasoning step: {answer}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"""

REWRITE_FORMAT_STR = """<|start_header_id|>user<|end_header_id|>\nQuestion: {question}\nThe weak answer: {answer}\nReflection: {review}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"""

SEP = None