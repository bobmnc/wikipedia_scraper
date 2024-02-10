import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
import logging
import torch
from nltk import sent_tokenize
# import nltk
# nltk.download('punkt')

def scrape_wikipedia_article(url : str,tokenizer : AutoTokenizer):
    '''
    Scrape a wikipedia article based on url

    inputs :
        - url : (str) an url to a wikipedia article
        - tokenizer (AutoTokenizer) : huggingface tokenizer to tokenize text
    Outputs :
        - article_text (str) the text with line skip between paragraphs
        - tokenized_text (tensor) tokenized text sentence by sentence
        - attention_masks (tensor) attentions mask that takes into account padding
    '''
    # Send a GET request to the Wikipedia URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the main content element of the Wikipedia article
        content = soup.find(id='mw-content-text')
        paragraphs = content.find_all('p')
        article_text = ''
        sentences = []
        tokenized_text = []
        attention_masks = []
        for paragraph in paragraphs:
            # Exclude refs
            for tag in paragraph.find_all('a'):
                tag.replace_with(tag.text)

            # Exclude citations 
            for sup in paragraph.find_all('sup'):
                sup.extract()
            # Exclude bold text
            for tag in paragraph.find_all('b'):
                tag.replace_with(tag.text)

            article_text += paragraph.text + '\n'

            sentences.extend(sent_tokenize(paragraph.text.strip()))
        batch_size = 8  
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:min(i + batch_size,
                                              len(sentences))]
            encoded_sentences = tokenizer(batch_sentences,
                                          padding='max_length', 
                                          truncation=True, 
                                          return_tensors='pt')
            tokenized_text.extend(encoded_sentences['input_ids'])
            attention_masks.extend(encoded_sentences['attention_mask'])

        tokenized_text = torch.stack(tokenized_text)
        attention_masks = torch.stack(attention_masks)
        return article_text,tokenized_text,attention_masks
    else:
        # If the request was not successful, print an error message
        logging.warning('Error: Unable to retrieve the Wikipedia article.')
        return None,None,None

# Example usage:
if __name__ == "__main__":
    # URL of the Wikipedia article to scrape
    wikipedia_url = 'https://en.wikipedia.org/wiki/Natural_language_processing'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    # Call the scrape_wikipedia_article function
    article_content,tokenized_text,attention_masks = scrape_wikipedia_article(wikipedia_url,tokenizer)
    
    if article_content:
        print(article_content)
