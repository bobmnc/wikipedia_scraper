from constants import categories
from get_urls_interest import get_urls
from scraper_soup import scrape_wikipedia_article
import h5py
import os
from os.path import join,isdir
import torch
from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized_dir = '../data/tokenized_data'
    list_urls,categories_urls = get_urls(categories=categories)
    f = h5py.File('../data/text_data/text_dataset.h5', 'w')
    
    
    for category in categories:
        # Create a group for each category
        if not isdir(join(tokenized_dir,category)):
            os.mkdir(join(tokenized_dir,category))
        group = f.create_group(category)
        
        # Get the indices of URLs corresponding to the current category
        indices = [i for i, cat in enumerate(categories_urls) if cat == category]
        
        # Create a chunked dataset for outputs
        dset = group.create_dataset('text', shape=(len(indices),),
                                     dtype=h5py.special_dtype(vlen=str),
                                     chunks=True)
        
        for i, idx in enumerate(indices):
            text,tokenized_text,attention_masks = scrape_wikipedia_article(url=list_urls[idx],
                                                       tokenizer=tokenizer)
            data_dict = {
                    'tokenized_text': tokenized_text,
                    'attention_masks': attention_masks
                }
            torch.save(data_dict,join(tokenized_dir,category,str(i))+'.pt')

            if text is not None:
                dset[i] = text
            else:
                # Handle case where fetching data fails
                dset[i] = 'Request failed'



if __name__=='__main__':
    main()
            
