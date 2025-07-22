# Common
COT_TASK_DESC = """<|im_start|>system\nPlease reason step by step, and put your final answer within \\\\boxed{{}}.<|im_end|>"""

# GPQA 
# COT_TASK_DESC = """<|im_start|>system\nPlease reason step by step, and put your final answer within \\\\boxed{{$LETTER}}(without quotes) where LETTER is one of ABCD.<|im_end|>"""

# step-level 
CRITIQUE_TASK_DESC = "<|im_start|>system\nThere is a weak reasoning step in the solution, please provide a strict reflection to correct only this one step with less than 150 tokens. Don't output the complete solution.<|im_end|>" 

# Common
REWRITE_TASK_DESC = "<|im_start|>system\nPlease refine the weak answer according to your Reflection. Not allowed to use code to solve the question. Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>"

# GPQA 
# REWRITE_TASK_DESC = "<|im_start|>system\nPlease refine the weak answer according to your Reflection. Not allowed to use code to solve the question. Please reason step by step, and put your final answer within \\boxed{{$LETTER}}(without quotes) where LETTER is one of ABCD.<|im_end|>"


COT_FORMAT_STR = (
    """<|im_start|>user\n\n{question}<|im_end|>\n<|im_start|>assistant\n"""
)

CRITIQUE_FORMAT_STR = """<|im_start|>user\nQuestion: {question}\nThe weak reasoning step: {answer}<|im_end|>\n<|im_start|>assistant\n"""

REWRITE_FORMAT_STR = """<|im_start|>user\nQuestion: {question}\nThe weak answer: {answer}\nReflection: {review}<|im_end|>\n<|im_start|>assistant\n"""

SEP = None