from transformers import AutoTokenizer,AutoModelForSequenceClassification
import logging
import torch


def filter_language(text : list[str],list_languages : list[str],
                    is_tokenized : bool =False,limit_prob :float = 0.9):
    '''
    filter the texts if they are in a predefined list of languages

    Inputs :
        - text (list(str)) : a list of text to filter
        - list_languages (list(str)) a list of languages to keep
            must be a subset of : 
        - is_tokenized (bool) : if the text is already tokenized
        - limit_prob (float) : limit confidence in the language
    Outputs :
        - idx_filtered (tensor) a list of the idx to keep according 
        to the detected languages
        
    '''
    if is_tokenized:
        logging.warning('The use of tokenized text is not supported \
                         for the moment (compatibility issues)')
        return
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    if len(inputs.input_ids.shape)<=1:
        inputs.input_ids = inputs.input_ids.view(1,-1)
    with torch.no_grad():
        logits = model(**inputs).logits

    preds = torch.softmax(logits, dim=-1)

    id2lang = model.config.id2label
    filtered_preds = preds > limit_prob

    # Step 2: Get the indices of values greater than limit_prob
    idxs = torch.nonzero(filtered_preds, as_tuple=True)
    idx_filtered = []
    for i in idxs:
        if id2lang[i.item()] in list_languages:
            idx_filtered.append(i.item())

    idx_filtered = torch.tensor(idx_filtered)
    return idx_filtered

print(filter_language('je mange du chocolat',['fr']))