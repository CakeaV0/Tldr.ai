import requests
import os  # For future key management
from summarizer import summarize_article
from topic_modeler import group_articles_by_topic

# --- Configuration ---
API_KEY = '3850b9fca529486cbbc5fdc0b6553b38' 
COUNTRY = 'us'
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'

# --- Function to Fetch News ---
def fetch_top_headlines(api_key, country_code, category=None):
    """
    Fetches top news headlines from NewsAPI for a given country.
    """
    params = {
        'country': country_code,
        'apiKey': api_key
    }
    if category:
        params['category'] = category

    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        news_data = response.json()

        if news_data['status'] == 'ok':
            return news_data['articles']
        else:
            print(f"API Error: {news_data.get('message')}")
            print("Full response:", news_data)
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return None

# --- Main ---
if __name__ == "__main__":
    print(f"Fetching top headlines from {COUNTRY}...")
    articles = fetch_top_headlines(API_KEY, COUNTRY)

    if articles:
        print(f"Successfully fetched {len(articles)} articles.")
        print("-" * 30)

        # Group articles by topic using embeddings
        grouped_articles = group_articles_by_topic(articles, num_clusters=3)
        grouped_articles.sort(key=lambda x: x[0])  # Sort by cluster ID

        current_cluster = -1
        for cluster_id, article in grouped_articles:
            if cluster_id != current_cluster:
                print("\n" + "=" * 50)
                print(f"🧠 Topic Cluster {cluster_id + 1}")
                print("=" * 50)
                current_cluster = cluster_id

            print(f"• {article['title']}")
            print(f"  Source: {article['source']['name']}")
            print(f"  URL: {article['url']}")

            article_text = article.get('content') or article.get('description') or ""
            if article_text:
                summary = summarize_article(article_text)
                print(f"  Summary: {summary}")
            else:
                print("  Summary: [No content available]")

            print()
    else:
        print("Failed to fetch articles.")
