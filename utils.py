from itertools import chain
import string
from transformers import StoppingCriteria
from subprocess import check_output

def get_gpu_type():
    res = check_output(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    res = res.decode('utf-8')
    gpu = res.split('\n')[1].split(', ')[1]
    return gpu

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids, scores):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if stop == last_token:
                return True
        return False

####################### PROMPT CONSTANTS ################################

WORD_STARTS = ["▁", "Ċ", "Ġ", "<0x0A>"]


# RACE

RACE_POST_PROMPT = "\n\nGiven the above passage, please answer the following multiple choice question with either (A), (B), (C), or (D).\n\n{question}\n\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(C) {options[3]}"

RACE_GEN = "("


# DISCRIM EVAL

DISCRIM_POST_PROMPT = "\n\nPlease answer the above question with either \"yes\" or \"no\"."

DISCRIM_GEN = "Based on the information provided if I had to choose between \"yes\" and \"no\" my answer would be \""


# SUMMARIZATION

SUM_POST_PROMPT = """

Summarize the above text as concisely and accurately as possible. Make sure to include all the information throughout the text."""


# SENTIMENT

SENT_POST_PROMPT = "\n\nWhat is the sentiment of the above movie review? Please answer ONLY with either \"positive\" or \"negative\"."

SENT_GEN = "The sentiment of the above movie review is \""


# DROP

DROP_POST_PROMPT = "\n\n{question}\n\nAnswer the above question, using information from the passage. Give ONLY the answer without any extra information."


# [PRE_PROMPT, CONTENT, POST_PROMPT, GEN_START]
PROMPT_MAP = {
    #'cais/mmlu': [MMLU_PROMPT, MMLU_IC_PROMPT, ""],
    #'bigbio/med_qa': [MED_QA_PROMPT, "", ""],
    'Anthropic/discrim-eval': ["", "{filled_template}", DISCRIM_POST_PROMPT, DISCRIM_GEN],
    'EdinburghNLP/xsum': ["", "{document}", SUM_POST_PROMPT, ""],
    'cnn_dailymail': ["", "{article}", SUM_POST_PROMPT, ""],
    'stanfordnlp/imdb': ["", "{text}", SENT_POST_PROMPT, SENT_GEN],
    'ucinlp/drop': ["", "{passage}", DROP_POST_PROMPT, ""],
    'ehovy/race': ['', "{article}", RACE_POST_PROMPT, RACE_GEN],
}

TEMPLATE_MAP = {
    "Mistral-7B-Instruct-v0.1": ["[INST] ", " [/INST]"],
    "Meta-Llama-3-8B-Instruct": ["<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"]
}

EOS_TOK_MAP = {
    "Meta-Llama-3-8B-Instruct": "<|eot_id|>",
}

################################################################################
