#!/usr/bin/env python3
"""Auto-generate simple RAG training examples from docs/*.txt and docs/*.md.
Writes docs/rag_training_examples_expanded.jsonl

Heuristics:
- Split documents into sentences.
- Keep sentences matching health-related keywords.
- Map keywords to question templates when possible.
- Use the matching sentence (and optional surrounding sentence) as the answer.
- Include source filename in Sources and include a short context.
This is a bootstrap generator; review outputs before using for fine-tuning.
"""
import os
import re
import json
from pathlib import Path

DOCS_DIR = Path('docs')
OUT_FILE = DOCS_DIR / 'rag_training_examples_expanded.jsonl'
SEED_FILE = DOCS_DIR / 'rag_training_examples.jsonl'

# keywords and mapping to nicer question templates
KEYWORD_TEMPLATES = [
    ('race', 'Does race have an impact on breast cancer detection?'),
    ('ethnic', 'Does ethnicity affect breast cancer detection or outcomes?'),
    ('screen', 'What does the document say about breast cancer screening?'),
    ('mammog', 'What is a mammogram and why is it done?'),
    ('dense', 'Can dense breasts affect mammogram accuracy?'),
    ('symptom', 'What are the signs of breast cancer I should watch for?'),
    ('sign', 'What are the signs of breast cancer I should watch for?'),
    ('risk', 'What does the document say about risk factors for breast cancer?'),
    ('access', 'How does access to care affect breast cancer outcomes?'),
    ('diagnos', 'Does finding breast cancer earlier change survival?'),
    ('treatment', 'How are breast cancers commonly treated?'),
    ('screening', 'What does the document say about breast cancer screening?'),
    ('ultrasound', 'When is ultrasound or MRI recommended in addition to mammography?'),
    ('MRI', 'When is MRI recommended for breast imaging?'),
]

SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')


def split_sentences(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    parts = SENTENCE_SPLIT_RE.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def find_keyword_template(sentence):
    s = sentence.lower()
    for key, template in KEYWORD_TEMPLATES:
        if key in s:
            return key, template
    return None, None


def build_examples_from_file(path, max_per_file=25):
    text = ''
    try:
        text = path.read_text(encoding='utf-8')
    except Exception:
        try:
            text = path.read_text(encoding='latin-1')
        except Exception:
            return []
    sentences = split_sentences(text)
    examples = []
    used_idxs = set()
    for i, sent in enumerate(sentences):
        if len(examples) >= max_per_file:
            break
        key, q_template = find_keyword_template(sent)
        if not key:
            continue
        # take the sentence and one sentence before/after as context when available
        context_sentences = []
        if i > 0:
            context_sentences.append(sentences[i-1])
        context_sentences.append(sent)
        if i+1 < len(sentences):
            context_sentences.append(sentences[i+1])
        context = '\n\n'.join(context_sentences)
        # craft answer: prefer the matched sentence and make sure ends with period
        answer = sent.strip()
        if not answer.endswith('.') and not answer.endswith('?') and not answer.endswith('!'):
            answer = answer + '.'
        # map to nicer question if available else generic
        if q_template:
            question = q_template
        else:
            # fallback generic question
            question = f"What does the document say about {key}?"
        example = {
            'context': f"{context} (Source: {path.as_posix()})",
            'question': question,
            'answer': f"{answer} Sources: [{path.name}]"
        }
        examples.append(example)
    return examples


def main():
    all_examples = []
    # include seed examples if present
    if SEED_FILE.exists():
        try:
            with SEED_FILE.open('r', encoding='utf-8') as fh:
                for line in fh:
                    line=line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        all_examples.append(obj)
                    except Exception:
                        continue
        except Exception:
            pass

    # walk docs for txt/md files
    files = list(DOCS_DIR.rglob('*.txt')) + list(DOCS_DIR.rglob('*.md'))
    files = sorted(files)
    for f in files:
        ex = build_examples_from_file(f, max_per_file=30)
        if ex:
            all_examples.extend(ex)

    # deduplicate by question+answer
    seen = set()
    uniq = []
    for e in all_examples:
        key = (e.get('question','').strip(), e.get('answer','').strip())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)

    # limit total size
    MAX_TOTAL = 1000
    uniq = uniq[:MAX_TOTAL]

    # write out
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open('w', encoding='utf-8') as out:
        for e in uniq:
            out.write(json.dumps(e, ensure_ascii=False) + '\n')

    print(f'Wrote {len(uniq)} examples to {OUT_FILE}')


if __name__ == '__main__':
    main()
