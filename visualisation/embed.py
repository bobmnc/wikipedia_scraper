import os
from os.path import join
import torch
from transformers import AutoModel
from tqdm import tqdm


def embed_dataset(data_dir : str,embedding_model : str = 'distilbert-base-uncased'):
    categories = os.listdir(data_dir)
    model = AutoModel.from_pretrained(embedding_model)
    list_embeds = []
    labels = []
    for category in tqdm(categories) :
        ### to do : batched processing
        for text_file in os.listdir(join(data_dir,category)):
            text = torch.load(join(data_dir,
                                   category,
                                   text_file))
            print(text.shape)
            with torch.no_grad():
                embedding = model(text)
            list_embeds.append(embedding.last_hidden_state[:,0].numpy())
            labels.append(category)
    return list_embeds,labels

if __name__=='__main__':
    embed_dataset('../data/tokenized_data')


