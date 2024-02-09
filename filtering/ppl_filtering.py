from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

def filter_text(text,is_tokenized=False,limit_ppl=100):
    is_batched = len(text.shape)>1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not is_tokenized:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
        tokenized_text = tokenizer(text)
    else :
        tokenized_text = text
    if not is_batched:
        tokenized_text = tokenized_text.view(1,-1)
    model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)
    with torch.no_grad():
        outputs = model(tokenized_text,labels = tokenized_text[:,1:] )
    loss = outputs.loss
    ppl = torch.exp(loss)
    to_keep = (ppl<=limit_ppl)
    return ppl,to_keep