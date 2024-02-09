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
        - tokenized_text (list[tensors]) tokenized text paragraph by paragraph
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
        tokenized_text = []

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
            for sentence in sent_tokenize(paragraph.text.strip()):
                tokenized_text.append(tokenizer.encode(sentence.strip().lower(),
                                    max_length = 512,
                                    padding='max_length',
                                    truncation=True))
        tokenized_text = torch.tensor(tokenized_text)
        return article_text,tokenized_text
    else:
        # If the request was not successful, print an error message
        logging.warning('Error: Unable to retrieve the Wikipedia article.')
        return None,None

# Example usage:
if __name__ == "__main__":
    # URL of the Wikipedia article to scrape
    wikipedia_url = 'https://en.wikipedia.org/wiki/Natural_language_processing'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    # Call the scrape_wikipedia_article function
    article_content,tokenized_text = scrape_wikipedia_article(wikipedia_url,tokenizer)
    
    if article_content:
        print(article_content)
