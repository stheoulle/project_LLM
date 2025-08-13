"""Runtime RAG utilities with optional cross-encoder reranking and generator prompting.

Functions:
- answer_question(question, top_k=4, generator_dir='docs/rag_generator') -> dict

Behavior:
- Retrieve candidate passages from local indexer.
- Optionally rerank with a CrossEncoder from sentence-transformers (if installed) for better precision.
- If a generator model exists, call it with a grounded prompt that includes the top passages and an instruction to cite sources and avoid hallucination.
- Otherwise produce an extractive synthesis and cite sources.
"""
from typing import List, Dict, Any, Optional
import os
import json
import numpy as np

# local indexer provider
from app import llm


def _format_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for h in hits:
        if isinstance(h, dict):
            text = h.get('text','')
            path = h.get('path') or h.get('filepath') or h.get('source') or None
            score = h.get('score')
        else:
            text = str(h)
            path = None
            score = None
        full = text.replace('\n', ' ').strip()
        if not full:
            continue
        out.append({'full': full, 'path': path, 'score': score})
    return out


def _apply_cross_encoder(question: str, candidates: List[Dict[str, Any]], top_k: int = 5):
    """Rerank candidates using a CrossEncoder(question, passage) if available. Returns top_k candidates sorted."""
    try:
        from sentence_transformers import CrossEncoder
        texts = [(question, c['full']) for c in candidates]
        # try a standard cross-encoder model name
        for model_name in ['cross-encoder/ms-marco-MiniLM-L-6-v2', 'cross-encoder/stsb-roberta-large']:
            try:
                ce = CrossEncoder(model_name)
                scores = ce.predict([ (q,p) for q,p in texts ])
                break
            except Exception:
                ce = None
                scores = None
        if scores is None:
            # try generic small model instantiation
            ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            scores = ce.predict([ (q,p) for q,p in texts ])

        # attach scores and sort
        for i, c in enumerate(candidates):
            c['rerank_score'] = float(scores[i])
        candidates_sorted = sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
        return candidates_sorted[:top_k]
    except Exception:
        # cross-encoder not available — return top_k by original order
        return candidates[:top_k]


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Chunk text into overlapping windows of approx chunk_size characters."""
    if not text:
        return []
    text = text.replace('\n', ' ')
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == L:
            break
        start = end - overlap if (end - overlap) > start else end
    return chunks


def build_passage_corpus(docs_dir: str = 'docs', chunk_size: int = 400, overlap: int = 50):
    """Read all .txt/.md files under docs_dir and produce a list of passages with metadata."""
    passages = []  # each: {'text':..., 'source':path, 'offset':int}
    for root, _, files in os.walk(docs_dir):
        for f in files:
            if f.lower().endswith('.txt') or f.lower().endswith('.md'):
                full = os.path.join(root, f)
                try:
                    txt = open(full, 'r', encoding='utf-8').read()
                except Exception:
                    continue
                chs = chunk_text(txt, chunk_size=chunk_size, overlap=overlap)
                for i, c in enumerate(chs):
                    passages.append({'text': c, 'source': full, 'idx': i})
    return passages


def embed_passages_and_save(passages: List[Dict[str, Any]], out_path: str = 'docs/passages_embeddings.npz'):
    """Embed passages using sentence-transformers if available, else TF-IDF fallback, save a .npz with arrays.
    Returns dict with paths and counts.
    """
    texts = [p['text'] for p in passages]
    sources = [p['source'] for p in passages]

    emb = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb = model.encode(texts, convert_to_numpy=True)
        method = 'sbert'
    except Exception:
        # fallback TF-IDF
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=20000, stop_words='english')
            mat = vec.fit_transform(texts)
            # convert to dense (might be large but passages small)
            emb = mat.toarray()
            method = 'tfidf'
        except Exception as e:
            raise RuntimeError(f"No embedding backend available: {e}")

    # Save embeddings and metadata
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez_compressed(out_path, embeddings=emb, sources=np.array(sources, dtype=object), texts=np.array(texts, dtype=object))
    meta = {'n_passages': len(texts), 'method': method, 'path': out_path}
    with open(out_path + '.meta.json', 'w', encoding='utf-8') as mf:
        json.dump(meta, mf)
    return meta


def build_passage_index(passages_npz: str = 'docs/passages_embeddings.npz', index_path: str = 'docs/passages_index'):
    """Build a FAISS or sklearn index over passage embeddings saved in passages_npz. Returns meta dict."""
    if not os.path.exists(passages_npz):
        raise FileNotFoundError(passages_npz)
    arr = np.load(passages_npz, allow_pickle=True)
    emb = arr['embeddings']
    sources = arr['sources'].tolist() if 'sources' in getattr(arr, 'files', []) else []

    n, d = emb.shape
    # try faiss
    try:
        import faiss
        idx = faiss.IndexFlatL2(d)
        idx.add(emb.astype(np.float32))
        faiss.write_index(idx, index_path + '.faiss')
        meta = {'index_type': 'faiss', 'index_file': index_path + '.faiss', 'n': n}
        with open(index_path + '.meta.json', 'w', encoding='utf-8') as mf:
            json.dump({'sources': sources, **meta}, mf)
        return {'index': index_path + '.faiss', 'n': n}
    except Exception:
        # sklearn fallback
        try:
            from sklearn.neighbors import NearestNeighbors
            import pickle
            nn = NearestNeighbors(n_neighbors=min(10, n), algorithm='auto').fit(emb)
            with open(index_path + '.pkl', 'wb') as fh:
                pickle.dump({'nn': nn, 'sources': sources}, fh)
            return {'index': index_path + '.pkl', 'n': n}
        except Exception as e:
            raise RuntimeError(f"Failed to build passage index: {e}")


_original_answer_question = None
try:
    _original_answer_question = globals().get('answer_question')
except Exception:
    _original_answer_question = None


def answer_question(question: str, top_k: int = 4, generator_dir: str = 'docs/rag_generator') -> Dict[str, Any]:
    """Prefer passage-level retrieval + reranking + generator.
    If passages index not present, fall back to document-level retrieval.
    """
    passages_npz = 'docs/passages_embeddings.npz'
    index_meta = None
    # if passages exist, query passages
    if os.path.exists(passages_npz):
        try:
            arr = np.load(passages_npz, allow_pickle=True)
            emb = arr['embeddings']
            texts = arr['texts'].tolist() if 'texts' in getattr(arr, 'files', []) else []
            sources = arr['sources'].tolist() if 'sources' in getattr(arr, 'files', []) else []

            # try faiss index
            idx_file = 'docs/passages_index.faiss'
            if os.path.exists(idx_file):
                try:
                    import faiss
                    index = faiss.read_index(idx_file)
                    q = None
                    # embed query using indexer embedding model
                    idxer = llm.get_indexer('docs')
                    if getattr(idxer, 'embedding_model', None) is not None:
                        q = idxer.embedding_model.encode([question], convert_to_numpy=True)[0].astype(np.float32)
                    else:
                        # TF-IDF fallback: vectorize question using indexer's vectorizer then pad/convert
                        if getattr(idxer, 'vectorizer', None) is not None:
                            q = idxer.vectorizer.transform([question]).toarray()[0].astype(np.float32)

                    if q is None:
                        # fallback to document-level retriever
                        raise RuntimeError('No embedding model available to query passage index')

                    D, I = index.search(np.expand_dims(q,axis=0), top_k)
                    hits = []
                    for i in I[0]:
                        hits.append({'text': texts[i], 'path': sources[i] if i < len(sources) else None, 'score': None})
                    # proceed to rerank/generate using same logic as above
                    # reuse earlier logic by calling internal rerank/generate helpers
                    # For simplicity, call the earlier implemented answer_question logic by temporarily swapping
                    # But to avoid recursion, use current file's reranker/generator sections
                    # We'll reuse helper: _apply_cross_encoder and generation logic
                    # Convert hits to candidates
                    from sentence_transformers import CrossEncoder
                    candidates = []
                    for h in hits:
                        candidates.append({'full': h['text'], 'path': h.get('path'), 'score': h.get('score')})
                    # rerank
                    try:
                        from sentence_transformers import CrossEncoder
                        ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                        pairs = [(question, c['full']) for c in candidates]
                        scores = ce.predict(pairs)
                        for i, c in enumerate(candidates):
                            c['rerank_score'] = float(scores[i])
                        candidates_sorted = sorted(candidates, key=lambda x: x.get('rerank_score',0), reverse=True)[:top_k]
                    except Exception:
                        candidates_sorted = candidates[:top_k]

                    # try generator
                    try:
                        gen_dir = os.path.join('docs','rag_generator')
                        if os.path.exists(gen_dir):
                            from transformers import T5ForConditionalGeneration, T5TokenizerFast
                            import torch
                            tokenizer = T5TokenizerFast.from_pretrained(gen_dir)
                            model = T5ForConditionalGeneration.from_pretrained(gen_dir)
                            prompt_ctx = '\n\n'.join([f"[{i+1}] {c['full']} (source: {os.path.basename(c.get('path') or 'doc')})" for i,c in enumerate(candidates_sorted)])
                            prompt = f"Answer using only the contexts below. Cite sources.\n\nCONTEXTS:\n{prompt_ctx}\n\nQUESTION: {question}\n\nAnswer:"
                            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                            with torch.no_grad():
                                out = model.generate(**inputs, max_length=256, num_beams=4)
                            gen = tokenizer.decode(out[0], skip_special_tokens=True)
                            return {'answer': gen, 'sources':[os.path.basename(c.get('path') or 'doc') for c in candidates_sorted], 'snippets': candidates_sorted, 'used_generator': True, 'used_reranker': True}
                    except Exception:
                        # fallback extractive
                        lines = [f"- {os.path.basename(c.get('path') or 'doc')}: {c['full'][:300]}" for c in candidates_sorted]
                        return {'answer': '\n'.join(lines), 'sources':[os.path.basename(c.get('path') or 'doc') for c in candidates_sorted], 'snippets': candidates_sorted, 'used_generator': False, 'used_reranker': True}

                except Exception:
                    pass
        except Exception:
            pass

    # fallback to document-level answer
    # reuse earlier implementation
    return _original_answer_question(question, top_k=top_k, generator_dir=generator_dir) if _original_answer_question else {'answer': 'No retrieval method available', 'sources': [], 'snippets': [], 'used_generator': False, 'used_reranker': False}
