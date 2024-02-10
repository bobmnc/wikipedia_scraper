import os
from os.path import join
import torch
from transformers import AutoModel
from tqdm import tqdm
import numpy as np


def embed_dataset(data_dir : str,embedding_model : str = 'distilbert-base-uncased'):
    categories = os.listdir(data_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(embedding_model)
    model = model.to(device)
    list_embeds = []
    labels = []
    for category in tqdm(categories[:5]) :
        ### to do : batched processing
        for text_file in os.listdir(join(data_dir,category)):
            data = torch.load(join(data_dir,
                                   category,
                                   text_file))
            
            input_ids = data['tokenized_text'].to(device).reshape(-1,512)
            attention_mask = data['attention_masks'].to(device).reshape(-1,512)
            with torch.no_grad():
                embedding = model(input_ids,
                                  attention_mask=attention_mask)
            list_embeds.append(embedding.last_hidden_state[:,0].cpu().numpy())
            labels+=[category for k in range(input_ids.shape[0])]
    list_embeds = np.concatenate(list_embeds)
    return list_embeds,labels

## test of the function
if __name__=='__main__':
    embed_dataset('../data/tokenized_data')


