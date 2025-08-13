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

# Generator cache (lazy loaded)
_GEN_MODEL = None
_GEN_TOKENIZER = None
_GEN_DEVICE = None
# Debug/log variables
_GEN_LOAD_ERROR = None
_GEN_LOAD_TRACE = None
_GEN_LAST_GENERATION_ERROR = None
_GEN_LAST_GENERATION_TRACE = None


def _load_generator(generator_dir: str = 'docs/rag_generator'):
    """Lazy load and cache generator model/tokenizer. Returns (model, tokenizer, device) or (None,None,None) on failure.
    Debug info stored in module variables for inspection.
    """
    global _GEN_MODEL, _GEN_TOKENIZER, _GEN_DEVICE, _GEN_LOAD_ERROR, _GEN_LOAD_TRACE
    if _GEN_MODEL is not None and _GEN_TOKENIZER is not None:
        return _GEN_MODEL, _GEN_TOKENIZER, _GEN_DEVICE
    try:
        # import lazily to avoid import-time overhead
        from transformers import T5ForConditionalGeneration, T5TokenizerFast
        import torch, sys, traceback
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(generator_dir):
            _GEN_LOAD_ERROR = f"generator dir not found: {generator_dir}"
            _GEN_LOAD_TRACE = None
            print(f"[rag::_load_generator] { _GEN_LOAD_ERROR }", file=sys.stderr)
            return None, None, None
        tokenizer = T5TokenizerFast.from_pretrained(generator_dir)
        model = T5ForConditionalGeneration.from_pretrained(generator_dir).to(device)
        _GEN_MODEL = model
        _GEN_TOKENIZER = tokenizer
        _GEN_DEVICE = device
        _GEN_LOAD_ERROR = None
        _GEN_LOAD_TRACE = None
        print(f"[rag::_load_generator] loaded generator from {generator_dir} on device={device}", file=sys.stderr)
        return _GEN_MODEL, _GEN_TOKENIZER, _GEN_DEVICE
    except Exception as e:
        import sys, traceback
        _GEN_LOAD_ERROR = str(e)
        _GEN_LOAD_TRACE = traceback.format_exc()
        # print to stderr for debug monitoring
        print(f"[rag::_load_generator] ERROR loading generator: {e}", file=sys.stderr)
        print(_GEN_LOAD_TRACE, file=sys.stderr)
        return None, None, None


def get_generator_debug():
    """Return current debug status for generator loader and last generation."""
    return {
        'loaded': _GEN_MODEL is not None and _GEN_TOKENIZER is not None,
        'device': str(_GEN_DEVICE) if _GEN_DEVICE is not None else None,
        'load_error': _GEN_LOAD_ERROR,
        'load_trace': _GEN_LOAD_TRACE,
        'last_gen_error': _GEN_LAST_GENERATION_ERROR,
        'last_gen_trace': _GEN_LAST_GENERATION_TRACE,
    }


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
    global _GEN_LAST_GENERATION_ERROR, _GEN_LAST_GENERATION_TRACE
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
                        # try generator — use cached loader
                        model, tokenizer, device = _load_generator(os.path.join('docs', 'rag_generator'))
                        if model is not None and tokenizer is not None:
                            import torch
                            prompt_ctx = '\n\n'.join([f"[{i+1}] {c['full']} (source: {os.path.basename(c.get('path') or 'doc')})" for i,c in enumerate(candidates_sorted)])
                            PROMPT = (
                                "You are an assistant that answers health questions in plain language for patients. "
                                "Use ONLY the information in the contexts below. If the contexts do not contain an answer, say you do not know and advise consulting a healthcare professional. "
                                "Answer concisely (1-3 short paragraphs) and list sources at the end in the format: Sources: [file1], [file2].\n\n"
                                "CONTEXTS:\n{contexts}\n\nQUESTION: {question}\n\nAnswer:"
                            )
                            prompt = PROMPT.format(contexts=prompt_ctx, question=question)

                            # Use helper to generate and retry with an alternative prompt if the model echoes the prompt
                            alt_prompt = (
                                "Answer the QUESTION using ONLY the numbered CONTEXTS below. Provide a short, direct answer (1-3 short paragraphs).\n\n"
                                "CONTEXTS:\n{contexts}\n\nQUESTION: {question}\n\nAnswer:"
                            ).format(contexts=prompt_ctx, question=question)

                            gen, used_alt = _generate_with_prompt_retry(model, tokenizer, device, prompt, alt_prompts=[alt_prompt])
                            if gen is not None:
                                return {'answer': gen, 'sources':[os.path.basename(c.get('path') or 'doc') for c in candidates_sorted], 'snippets': candidates_sorted, 'used_generator': True, 'used_reranker': True}

                            # if generation failed or looked like an echo, fall back to extractive
                            import sys, traceback
                            _GEN_LAST_GENERATION_TRACE = _GEN_LAST_GENERATION_TRACE or None
                            print(f"[rag::answer_question] generation failed or looked like prompt-echo: {_GEN_LAST_GENERATION_ERROR}", file=sys.stderr)
                            lines = [f"- {os.path.basename(c.get('path') or 'doc')}: {c['full'][:300]}" for c in candidates_sorted]
                            return {'answer': '\n'.join(lines), 'sources':[os.path.basename(c.get('path') or 'doc') for c in candidates_sorted], 'snippets': candidates_sorted, 'used_generator': False, 'used_reranker': True}
                    except Exception as e:
                        # log generation error
                        _GEN_LAST_GENERATION_ERROR = str(e)
                        _GEN_LAST_GENERATION_TRACE = traceback.format_exc()
                        print(f"[rag::answer_question] ERROR in generation: {_GEN_LAST_GENERATION_ERROR}", file=sys.stderr)
                        print(_GEN_LAST_GENERATION_TRACE, file=sys.stderr)
                        # fallback extractive
                        lines = [f"- {os.path.basename(c.get('path') or 'doc')}: {c['full'][:300]}" for c in candidates_sorted]
                        return {'answer': '\n'.join(lines), 'sources':[os.path.basename(c.get('path') or 'doc') for c in candidates_sorted], 'snippets': candidates_sorted, 'used_generator': False, 'used_reranker': True}

                except Exception:
                    pass
        except Exception:
            pass

    # fallback to document-level answer
    # reuse earlier implementation: if no passage index is present, try the document-level indexer
    try:
        idxer = llm.get_indexer('docs')
        hits = idxer.query(question, top_k=top_k) if idxer is not None else []
        if hits:
            candidates = []
            for h in hits:
                if isinstance(h, dict):
                    text = h.get('text','')
                    path = h.get('path') or h.get('filepath') or h.get('source') or None
                    score = h.get('score')
                else:
                    text = str(h); path = None; score = None
                full = text.replace('\n',' ').strip()
                if not full:
                    continue
                candidates.append({'full': full, 'path': path, 'score': score})

            # Try generator using document-level candidates when available
            try:
                gen_model, gen_tokenizer, gen_device = _load_generator(os.path.join('docs','rag_generator'))
                if gen_model is not None and gen_tokenizer is not None:
                    prompt_ctx = '\n\n'.join([f"[{i+1}] {c['full']} (source: {os.path.basename(c.get('path') or 'doc')})" for i,c in enumerate(candidates)])
                    PROMPT = (
                        "You are an assistant that answers health questions in plain language for patients. "
                        "Use ONLY the information in the contexts below. If the contexts do not contain an answer, say you do not know and advise consulting a healthcare professional. "
                        "Answer concisely (1-3 short paragraphs) and list sources at the end in the format: Sources: [file1], [file2].\n\n"
                        "CONTEXTS:\n{contexts}\n\nQUESTION: {question}\n\nAnswer:"
                    )
                    prompt = PROMPT.format(contexts=prompt_ctx, question=question)

                    try:
                        inputs = gen_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512, padding='longest')
                        inputs = {k: v.to(gen_device) for k, v in inputs.items()}
                        gen_kwargs = dict(max_length=150, num_beams=4, early_stopping=True, no_repeat_ngram_size=3, length_penalty=1.0)
                        import torch
                        with torch.no_grad():
                            out = gen_model.generate(**inputs, **gen_kwargs)
                        gen = gen_tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        # Post-process: if model echoed the prompt or returned empty, treat as failed generation and fall back
                        if not gen or gen.strip() == '':
                            # treat as failure — will fall back to extractive below
                            _GEN_LAST_GENERATION_ERROR = 'empty_generation'
                        else:
                            low = gen.lower()
                            # detect if model just returned the prompt/instructions unintentionally
                            if 'you are an assistant' in low or 'contexts:' in low[:200].lower() or 'answer:' in low[:200].lower() or gen.strip().startswith(prompt.split('\n')[0]):
                                # record debug note and fall back
                                _GEN_LAST_GENERATION_ERROR = 'generation_looks_like_prompt_or_echo'
                            else:
                                # success — return generator answer
                                return {'answer': gen, 'sources':[os.path.basename(c.get('path') or 'doc') for c in candidates], 'snippets': candidates, 'used_generator': True, 'used_reranker': False}
                    except Exception as e:
                        # record generation error for debugging and fall back to extractive
                        try:
                            import traceback
                            _GEN_LAST_GENERATION_ERROR = str(e)
                            _GEN_LAST_GENERATION_TRACE = traceback.format_exc()
                        except Exception:
                            _GEN_LAST_GENERATION_ERROR = str(e)
                        # fall through to extractive fallback
                        pass
            except Exception:
                # fall through to extractive fallback
                pass

            lines = [f"- {os.path.basename(c.get('path') or 'doc')}: {c.get('full')[:300]}" for c in candidates]
            answer = '\n'.join(lines)
            return {'answer': answer, 'sources': [os.path.basename(c.get('path') or 'doc') for c in candidates], 'snippets': candidates, 'used_generator': False, 'used_reranker': False}
        else:
            return {'answer': 'No relevant documents found in the local index.', 'sources': [], 'snippets': [], 'used_generator': False, 'used_reranker': False}
    except Exception as e:
        # last resort: return informative error
        return {'answer': f'No retrieval method available: {e}', 'sources': [], 'snippets': [], 'used_generator': False, 'used_reranker': False}


def _generate_with_prompt_retry(model, tokenizer, device, prompt, alt_prompts=None, gen_kwargs=None):
    """Generate text and retry when output looks like a prompt-echo.

    Enhancements:
    - Detects common prompt tokens like 'CONTEXTS:', 'Source:', 'Extracted:'.
    - Uses difflib.SequenceMatcher to measure overlap between prompt and generation.
    - If no alt_prompts provided, constructs a couple of automatic shorter prompts to try.
    Returns (generation_text or None, used_alt_flag or None).
    """
    import traceback
    import difflib
    global _GEN_LAST_GENERATION_ERROR, _GEN_LAST_GENERATION_TRACE
    try:
        import torch
        if gen_kwargs is None:
            gen_kwargs = dict(max_length=150, num_beams=4, early_stopping=True, no_repeat_ngram_size=3, length_penalty=1.0)

        def _looks_like_echo(text, prompt_text):
            """Return (bool, reason_str) indicating whether text likely echoes the prompt."""
            if not text:
                return True, 'empty_generation'
            low = text.lower()
            p_low = (prompt_text or '').lower()

            # quick keyword checks
            echo_keywords = ['you are an assistant', 'contexts:', 'answer:', 'source:', 'extracted:']
            for kw in echo_keywords:
                if kw in low:
                    return True, f'keyword_echo:{kw}'

            # if the generation begins with the beginning of the prompt
            pstart = p_low.strip()[:80]
            if pstart and low.startswith(pstart):
                return True, 'starts_with_prompt'

            # overlap ratio between prompt and generation
            try:
                ratio = difflib.SequenceMatcher(None, p_low, low).quick_ratio()
            except Exception:
                ratio = 0.0
            # threshold: moderate overlap indicates echoing instructions/context
            if ratio > 0.40:
                return True, f'overlap_ratio:{ratio:.2f}'

            return False, None

        # prepare alt prompts if none provided
        auto_alts = []
        if not alt_prompts:
            # alt 1: remove high-level instruction and ask directly to answer from contexts
            auto_alts.append(
                "Answer the QUESTION using ONLY the CONTEXTS listed below. Provide a concise answer (1-3 short paragraphs).\n\nCONTEXTS:\n{contexts}\n\nQUESTION: {question}\n\nAnswer:"
            )
            # alt 2: explicitly instruct NOT to repeat contexts/instructions
            auto_alts.append(
                "Using only the numbered contexts, give a brief direct answer. Do NOT repeat the contexts or instructions; return the answer only.\n\nCONTEXTS:\n{contexts}\n\nQUESTION: {question}\n\nAnswer:"
            )
        else:
            auto_alts = list(alt_prompts)

        # Try primary prompt
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512, padding='longest')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        gen = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

        is_echo, reason = _looks_like_echo(gen, prompt)
        if not is_echo:
            _GEN_LAST_GENERATION_ERROR = None
            _GEN_LAST_GENERATION_TRACE = None
            return gen, None

        # record initial echo reason
        _GEN_LAST_GENERATION_ERROR = f'generation_looks_like_echo:{reason}'

        # Try alt prompts (formatting requires contexts/question placeholders to already be filled by caller)
        for idx, alt in enumerate(auto_alts):
            try:
                inputs = tokenizer(alt, return_tensors='pt', truncation=True, max_length=512, padding='longest')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                gen2 = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                is_echo2, reason2 = _looks_like_echo(gen2, alt)
                if not is_echo2:
                    _GEN_LAST_GENERATION_ERROR = None
                    _GEN_LAST_GENERATION_TRACE = None
                    used_flag = 'used_alt' if alt_prompts else f'used_auto_alt_{idx}'
                    return gen2, used_flag
                # else record the most recent reason
                _GEN_LAST_GENERATION_ERROR = f'generation_looks_like_echo_after_alt:{reason2}'
            except Exception as e:
                _GEN_LAST_GENERATION_ERROR = str(e)
                _GEN_LAST_GENERATION_TRACE = traceback.format_exc()
                # try next alt
                continue

        # All attempts looked like echoes — keep the last recorded reason
        if _GEN_LAST_GENERATION_ERROR is None:
            _GEN_LAST_GENERATION_ERROR = 'generation_looks_like_prompt_or_echo_after_retries'
        return None, None
    except Exception as e:
        _GEN_LAST_GENERATION_ERROR = str(e)
        _GEN_LAST_GENERATION_TRACE = traceback.format_exc()
        return None, None
