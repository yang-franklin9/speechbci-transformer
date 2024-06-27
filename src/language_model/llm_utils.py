import transformers
from .config import device, llm_name

tokenizer = transformers.AutoTokenizer.from_pretrained(llm_name)
llm = transformers.AutoModelForCausalLM.from_pretrained(llm_name).to(device)
llm_embed = llm.get_input_embeddings()

llm.eval()
for p in llm.parameters():
    p.requires_grad = False

def strip_bos_eos(token_list):
    new_list = []
    for x in token_list:
        if (x != tokenizer.bos_token_id and x != tokenizer.eos_token_id):
            new_list.append(x)
    return new_list

def tokenize(str):
	return strip_bos_eos(tokenizer(str).input_ids)
