from transformers import AutoTokenizer

def load_default_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    return tokenizer