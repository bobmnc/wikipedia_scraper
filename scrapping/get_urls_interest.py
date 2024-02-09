import requests


def get_urls(categories : list[str]):
    all_article_urls = []
    categories_urls = []

    # Iterate over each category
    for category in categories:
        api_url = 'https://en.wikipedia.org/w/api.php'
        # Define parameters for the API request
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': category,
            'cmlimit': 5,#00,  # Number of articles to retrieve per category
            'format': 'json'
        }

        # Send a GET request to the Wikipedia API
        response = requests.get(api_url, params=params)
        data = response.json()

        # Extract article titles from the response
        article_titles = [page['title'] for page in data['query']['categorymembers']]

        # Construct URLs from the article titles
        base_url = 'https://en.wikipedia.org/wiki/'
        article_urls = [base_url + title.replace(' ', '_') for title in article_titles]

        all_article_urls.extend(article_urls)
        categories_urls.extend(category)

    return all_article_urls,categories_urls

if __name__=='__main__':
    urls = get_urls(['Category:Machine learning'])
    print(urls[:10])