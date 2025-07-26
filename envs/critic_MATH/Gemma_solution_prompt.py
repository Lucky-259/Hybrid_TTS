# Common
COT_TASK_DESC = """<bos><start_of_turn>user\nPlease reason step by step, and put your final answer within \\\\boxed{{}}.\n"""

# GPQA
# COT_TASK_DESC = """<bos><start_of_turn>user\nPlease reason step by step, and put your final answer within \\\\boxed{{$LETTER}}(without quotes) where LETTER is one of ABCD\n"""

# solution-level
CRITIQUE_TASK_DESC = "<bos><start_of_turn>user\nThere is a weak solution, please provide a strict relection to correct it.\n" 

# Common
REWRITE_TASK_DESC = "<bos><start_of_turn>user\nPlease refine the weak answer according to your Reflection. Not allowed to use code to solve the question. Please reason step by step, and put your final answer within \\boxed{{}}.\n"

# GPQA
# REWRITE_TASK_DESC = "<bos><start_of_turn>user\nPlease refine the weak answer according to your Reflection. Not allowed to use code to solve the question. Please reason step by step, and put your final answer within \\boxed{{$LETTER}}(without quotes) where LETTER is one of ABCD.\n"


COT_FORMAT_STR = (
    """{question}<end_of_turn>\n<start_of_turn>model\n"""
)

CRITIQUE_FORMAT_STR = """Question: {question}\nThe weak reasoning step: {answer}<end_of_turn>\n<start_of_turn>model\n"""

REWRITE_FORMAT_STR = """Question: {question}\nThe weak answer: {answer}\nReflection: {review}<end_of_turn>\n<start_of_turn>model\n"""

SEP = None