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


def answer_question(question: str, top_k: int = 4, generator_dir: str = 'docs/rag_generator') -> Dict[str, Any]:
    """Answer a question using retrieval + optional reranker + generator.

    Returns { 'answer': str, 'sources': [...], 'snippets': [...], 'used_generator': bool, 'used_reranker': bool }
    """
    idx = llm.get_indexer('docs')
    hits = idx.query(question, top_k=top_k*3)  # retrieve more for reranking

    candidates = _format_hits(hits)

    # If no candidates, return early
    if not candidates:
        return {'answer': 'No documents found in index.', 'sources': [], 'snippets': [], 'used_generator': False, 'used_reranker': False}

    # Rerank with cross-encoder if available
    used_reranker = False
    try:
        reranked = _apply_cross_encoder(question, candidates, top_k=top_k)
        used_reranker = True
    except Exception:
        reranked = candidates[:top_k]
        used_reranker = False

    # Prepare snippets list
    snippets = []
    for c in reranked:
        short = c['full']
        # get first two sentences or 300 chars
        sentences = [s.strip() for s in short.split('.') if s.strip()]
        if len(sentences) >= 2:
            snippet = sentences[0] + '. ' + sentences[1] + '.'
        else:
            snippet = short[:300]
        snippets.append({'source': os.path.basename(c['path']) if c.get('path') else 'doc', 'snippet': snippet, 'score': c.get('rerank_score', c.get('score'))})

    # Try generator model to produce grounded answer
    used_generator = False
    generated_answer = None
    try:
        from pathlib import Path
        gen_dir = Path(generator_dir)
        if gen_dir.exists() and any(gen_dir.iterdir()):
            try:
                from transformers import T5ForConditionalGeneration, T5TokenizerFast
                import torch
                tokenizer = T5TokenizerFast.from_pretrained(str(gen_dir))
                model = T5ForConditionalGeneration.from_pretrained(str(gen_dir))
                model.eval()

                # Build prompt: include numbered contexts and instruction
                ctxs = '\n\n'.join([f"[{i+1}] {s['snippet']} (source: {s['source']})" for i,s in enumerate(snippets)])
                prompt = (
                    "Answer the question using ONLY the information in the CONTEXTS below. Cite sources by their filename in square brackets. "
                    "If the contexts do not contain an answer, reply 'No evidence in the provided documents.'\n\n"
                    f"CONTEXTS:\n{ctxs}\n\nQUESTION: {question}\n\nAnswer:"
                )

                inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                with torch.no_grad():
                    out = model.generate(**inputs, max_length=256, num_beams=4)
                generated_answer = tokenizer.decode(out[0], skip_special_tokens=True)
                used_generator = True
            except Exception:
                used_generator = False
    except Exception:
        used_generator = False

    answer = generated_answer if used_generator and generated_answer else None
    if not answer:
        # fallback: extractive synthesis and list sources
        lines = ["I found the following relevant passages:"]
        for s in snippets:
            score_str = f" (score={s['score']:.3f})" if s.get('score') is not None else ''
            lines.append(f"- {s['source']}{score_str}: {s['snippet']}")
        synth = ' '.join([s['snippet'] for s in snippets])
        if len(synth) > 800:
            synth = synth[:800].rsplit(' ',1)[0] + '...'
        lines.append('\nSynthesis: ' + synth)
        answer = '\n'.join(lines)

    return {'answer': answer, 'sources': [s['source'] for s in snippets], 'snippets': snippets, 'used_generator': used_generator, 'used_reranker': used_reranker}
