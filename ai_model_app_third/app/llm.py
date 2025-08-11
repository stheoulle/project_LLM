import os
import glob
import json
from typing import List, Dict, Optional

# Try to import optional heavy deps; fallbacks will be used if unavailable
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class LocalIndexer:
    def __init__(self, doc_dir: str = "docs"):
        self.doc_dir = doc_dir
        self.docs: List[Dict] = []  # list of {'path':..., 'text':...}
        self.embedding_model = None
        self.embeddings = None
        self.vectorizer = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # small fast embedding model
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.embedding_model = None

    def load_documents(self):
        docs = []
        if not os.path.exists(self.doc_dir):
            # No docs directory found — return empty list
            self.docs = []
            return self.docs

        patterns = ["**/*.txt", "**/*.md"]
        for pat in patterns:
            for path in glob.glob(os.path.join(self.doc_dir, pat), recursive=True):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                        docs.append({"path": path, "text": text})
                except Exception:
                    continue
        self.docs = docs
        return self.docs

    def build_index(self):
        # Load docs
        self.load_documents()
        texts = [d["text"] for d in self.docs]

        if self.embedding_model is not None and len(texts) > 0:
            try:
                self.embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
                return
            except Exception:
                self.embeddings = None

        # Fallback: TF-IDF vectorizer
        if SKLEARN_AVAILABLE and len(texts) > 0:
            try:
                self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
                self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            except Exception:
                self.vectorizer = None

    def query(self, query_text: str, top_k: int = 3):
        """Return top_k documents and similarity scores"""
        if not self.docs:
            return []

        # If embeddings available, use them
        if self.embeddings is not None and SENTENCE_TRANSFORMERS_AVAILABLE:
            q_emb = self.embedding_model.encode([query_text], convert_to_numpy=True)[0]
            sims = (self.embeddings @ q_emb) / ( (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)) + 1e-8 )
            idxs = list(reversed(sims.argsort()))[:top_k]
            results = []
            for i in idxs:
                results.append({"path": self.docs[i]["path"], "text": self.docs[i]["text"], "score": float(sims[i])})
            return results

        # Else, if TF-IDF available
        if SKLEARN_AVAILABLE and self.vectorizer is not None:
            q_vec = self.vectorizer.transform([query_text])
            sims = cosine_similarity(self.tfidf_matrix, q_vec).flatten()
            idxs = list(reversed(sims.argsort()))[:top_k]
            results = []
            for i in idxs:
                results.append({"path": self.docs[i]["path"], "text": self.docs[i]["text"], "score": float(sims[i])})
            return results

        # No index available
        return []


_indexer: Optional[LocalIndexer] = None


def get_indexer(doc_dir: str = "docs") -> LocalIndexer:
    global _indexer
    if _indexer is None:
        _indexer = LocalIndexer(doc_dir=doc_dir)
        _indexer.build_index()
    return _indexer


# Heuristic suggestion helper (kept similar behavior as before)

def _heuristic_suggest(chat_history, has_tabular):
    texts = " ".join([c.get('text','') for c in chat_history if c.get('text')])
    n_images = sum(len(c.get('images', [])) for c in chat_history)

    suggested = []
    if n_images > 0:
        suggested.append('images')
    if has_tabular:
        suggested.append('meta')
    report_keywords = ['report', 'finding', 'diagnosis', 'conclusion', 'opinion', 'recommend']
    if any(k in texts.lower() for k in report_keywords) or len(texts.split()) > 10:
        suggested.append('reports')

    if 'images' in suggested and 'meta' in suggested and 'reports' in suggested:
        modality = 'images+meta+reports'
    elif 'images' in suggested and 'meta' in suggested:
        modality = 'images+meta'
    elif 'images' in suggested:
        modality = 'images'
    elif 'meta' in suggested:
        modality = 'images+meta'
    else:
        modality = 'images'

    explanation = (
        f"Heuristic suggestion based on inputs: {len(chat_history)} messages, {n_images} images, "
        f"tabular={has_tabular}. Recommended modality: {modality}."
    )
    return modality, explanation


def analyze_chat_and_suggest(chat_history, uploaded_tabular=None, docs_dir: str = "docs"):
    """
    Analyze chat history and local documents to suggest a modality and provide assistant text.
    Uses local retrieval over `docs_dir` (expects .txt/.md files) and a heuristic for modality selection.
    Returns: dict { 'modality': str, 'assistant_text': str }
    """
    has_tabular = uploaded_tabular is not None
    # Simple modality suggestion
    modality, explanation = _heuristic_suggest(chat_history, has_tabular)

    # Build assistant message: include retrieval results from local docs if available
    indexer = get_indexer(docs_dir)
    concat_text = "\n\n".join([m.get('text','') for m in chat_history if m.get('text')])
    query_text = concat_text if concat_text.strip() else "medical multimodal training guidance"

    retrieved = indexer.query(query_text, top_k=3)

    assistant_parts = ["Modality suggestion: " + modality, "Reasoning: " + explanation]
    if retrieved:
        assistant_parts.append("Relevant local documents found:")
        for r in retrieved:
            snippet = r['text'][:400].replace('\n', ' ').strip()
            assistant_parts.append(f"- {os.path.basename(r['path'])} (score={r['score']:.3f}): {snippet}...")
    else:
        assistant_parts.append("No local documents found in 'docs/' or indexing unavailable.")

    assistant_text = "\n\n".join(assistant_parts)

    return {"modality": modality, "assistant_text": assistant_text}
