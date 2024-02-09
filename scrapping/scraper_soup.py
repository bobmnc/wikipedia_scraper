import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
import logging

def scrape_wikipedia_article(url,tokenizer):
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
            print(paragraph)
            article_text += paragraph.text + '\n'
            tokenized_text += tokenizer(paragraph.text.strip().lower()) ## supposed to work on a batch of text
        return article_text,tokenized_text
    else:
        # If the request was not successful, print an error message
        logging.warning('Error: Unable to retrieve the Wikipedia article.')
        return None

# Example usage:
if __name__ == "__main__":
    # URL of the Wikipedia article to scrape
    wikipedia_url = 'https://en.wikipedia.org/wiki/Natural_language_processing'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    # Call the scrape_wikipedia_article function
    article_content,tokenized_text = scrape_wikipedia_article(wikipedia_url,tokenizer)

    # Print the extracted text
    if article_content:
        print(article_content)
