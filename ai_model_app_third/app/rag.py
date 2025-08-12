"""Runtime RAG utilities: retriever + optional generator.

Functions:
- answer_question(question, top_k=4, generator_dir='docs/rag_generator') -> dict

Behavior:
- Uses the local indexer (app.llm.get_indexer) to retrieve top_k contexts.
- If a generator model exists under generator_dir (T5 saved by Trainer), uses transformers to generate an answer conditioned on the retrieved contexts.
- Otherwise returns an extractive synthesis (concatenated snippets) plus sources.

This module keeps external deps optional and falls back gracefully when transformers/torch are not installed.
"""
from typing import List, Dict, Any, Optional
import os

# local indexer provider
from app import llm


def _format_retrieval_snippets(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parts = []
    for h in hits:
        # hits from indexer: may contain 'text' and 'path' and 'score'
        if isinstance(h, dict):
            text = h.get('text', '')
            path = h.get('path') or h.get('path', None)
            score = h.get('score', None)
        else:
            text = str(h)
            path = None
            score = None
        s = text.replace('\n', ' ').strip()
        if not s:
            continue
        sentences = [seg.strip() for seg in s.split('.') if seg.strip()]
        short = (sentences[0] + ('. ' + sentences[1] + '.') if len(sentences) > 1 else sentences[0]) if sentences else s[:300]
        parts.append({'source': os.path.basename(path) if path else 'doc', 'score': score, 'snippet': short, 'full': s, 'path': path})
    return parts


def answer_question(question: str, top_k: int = 4, generator_dir: str = 'docs/rag_generator') -> Dict[str, Any]:
    """Answer a question using retrieval + optional generator.

    Returns { 'answer': str, 'sources': [...], 'snippets': [...], 'used_generator': bool }
    """
    idx = llm.get_indexer('docs')
    hits = idx.query(question, top_k=top_k)
    snippets = _format_retrieval_snippets(hits)

    # Compose fallback extractive answer
    extractive = ' '.join([p['snippet'] for p in snippets])
    if len(extractive) > 1000:
        extractive = extractive[:1000].rsplit(' ', 1)[0] + '...'

    # Try generator if available
    use_generator = False
    generated_answer = None
    try:
        from pathlib import Path
        gen_dir = Path(generator_dir)
        if gen_dir.exists() and any(gen_dir.iterdir()):
            # attempt to load transformers model
            try:
                from transformers import T5ForConditionalGeneration, T5TokenizerFast
                import torch
                tokenizer = T5TokenizerFast.from_pretrained(str(gen_dir))
                model = T5ForConditionalGeneration.from_pretrained(str(gen_dir))
                model.eval()

                # Build prompt: include retrieved contexts
                contexts = '\n\n'.join([p['full'] for p in snippets])
                prompt = f"answer: Context: {contexts} Question: {question}"
                inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                with torch.no_grad():
                    out = model.generate(**inputs, max_length=256, num_beams=3)
                generated_answer = tokenizer.decode(out[0], skip_special_tokens=True)
                use_generator = True
            except Exception:
                use_generator = False
    except Exception:
        use_generator = False

    answer = generated_answer if use_generator and generated_answer else extractive
    return {
        'answer': answer,
        'sources': [p['source'] for p in snippets],
        'snippets': snippets,
        'used_generator': use_generator
    }
