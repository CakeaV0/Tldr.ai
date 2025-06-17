from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import pandas as pd

model = SentenceTransformer("all-MiniLM-L6-v2")

def group_articles_by_topic(articles, num_clusters=3):
    texts = [a.get("content") or a.get("description") or a.get("title", "") for a in articles]
    embeddings = model.encode(texts)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    tsne = TSNE(
    n_components=2,
    perplexity=min(30, max(2, len(embeddings) - 1)),  # Ensure perplexity < samples
    random_state=42
)

    reduced = tsne.fit_transform(embeddings)

    df = pd.DataFrame({
        "title": [a["title"] for a in articles],
        "url": [a["url"] for a in articles],
        "cluster": labels,
        "x": reduced[:, 0],
        "y": reduced[:, 1]
    })

    clustered_articles = list(zip(labels, articles))
    return clustered_articles, df
