from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import logging

def filter_text(text : list[str],is_tokenized : bool =False,limit_ppl : float=100.):
    '''
    return ppl computed with GPT2 on the batch of text and filter based on limit_ppl

    Inputs : 
        - text : list(str) the list of text to filter
        - is tokenized (bool) default = False : if the text is already tokenized
        - limit_ppl (float) default = 100. : limit perplexity for filtering
    Return:
        - ppl (tensor) : tensor of perplexity for the sentences
        - to keep (tensor[bool]) : is True if we want to keep the sentence 
    '''
    is_batched = len(text.shape)>1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not is_tokenized:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
        tokenized_text = tokenizer(text)
    else :
        logging.warning('TO DO : check compatibilty tokenizers')
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