import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.sparse import issparse

class SemanticIndex:
    def __init__(self, cards):
        self.cards = cards
        self.texts = [card['text'] for card in cards]
        self.vectorizer = TfidfVectorizer().fit(self.texts)
        self.tfidf_matrix = self.vectorizer.transform(self.texts)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)

    def retrieve(self, text, k=8):
        # BM25 (TF-IDF) retrieval
        tfidf_query = self.vectorizer.transform([text])
        tfidf_scores = self.tfidf_matrix @ tfidf_query.T
        if issparse(tfidf_scores):
            tfidf_scores = tfidf_scores.toarray().reshape(-1)
        else:
            tfidf_scores = np.asarray(tfidf_scores).reshape(-1)
        # Vector search
        query_emb = self.model.encode(text, convert_to_tensor=True)
        sim_scores = util.pytorch_cos_sim(query_emb, self.embeddings).cpu().numpy().reshape(-1)
        # Combine (simple sum)
        combined = tfidf_scores + sim_scores
        top_idx = np.argsort(combined)[::-1][:k]
        return [self.cards[i] for i in top_idx]

def main():
    with open('data/cards.json', 'r') as f:
        cards = json.load(f)
    index = SemanticIndex(cards)
    results = index.retrieve('amount in THB', k=3)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
