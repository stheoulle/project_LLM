#!/usr/bin/env python3
"""Validate, sample and CLEAN RAG training examples JSONL.

Usage examples:
  # validate and sample
  python scripts/validate_rag_examples.py --file docs/rag_training_examples_expanded.jsonl --sample 50 --out docs/rag_examples_report.csv

  # validate and create cleaned JSONL
  python scripts/validate_rag_examples.py --file docs/rag_training_examples_expanded.jsonl --clean --clean_out docs/rag_training_examples_cleaned.jsonl

The cleaner will:
- normalize source mentions to basenames in square brackets (e.g. [file.txt])
- strip common noise like 'Extracted: ... ---' segments
- ensure Answers contain a 'Sources:' token with bracketed basenames
- write cleaned JSONL to --clean_out
"""

import argparse
import json
import random
import csv
import os
import re
from pathlib import Path


def load_examples(path):
    examples = []
    with open(path, 'r', encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            line=line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append(obj)
            except Exception:
                # skip malformed
                continue
    return examples


def extract_sources(answer_or_context):
    # find bracketed filenames like [file.txt] or Source: path
    out = []
    s = answer_or_context or ''
    # simple bracket pattern
    for m in re.findall(r"\[([^\]]+)\]", s):
        out.append(m.strip())
    # also look for 'Source: path' or 'Sources: file1, file2'
    for m in re.findall(r"Source[s]?:\s*([^\n\[]+)", s, flags=re.IGNORECASE):
        parts = [p.strip().strip('.') for p in re.split(r'[;,|\\/]', m) if p.strip()]
        for p in parts:
            if p:
                out.append(p)
    # also try to capture filenames like word.pdf or word.txt
    for m in re.findall(r"([\w\-\. ]+\.(?:txt|pdf|md))", s, flags=re.IGNORECASE):
        out.append(m.strip())
    # normalize order and uniqueness
    seen = set()
    final = []
    for r in out:
        r2 = r.strip()
        if r2 and r2 not in seen:
            seen.add(r2)
            final.append(r2)
    return final


def normalize_basename(name):
    # take last token that looks like a filename
    name = (name or '').strip()
    # remove surrounding quotes or parentheses
    name = re.sub(r"^[\"'\(]+|[\"'\)]+$", '', name).strip()
    # split by whitespace and commas and take last segment if it contains a dot
    parts = re.split(r"[\s,]+", name)
    for p in reversed(parts):
        if '.' in p:
            return os.path.basename(p)
    # fallback to basename
    return os.path.basename(name)


def clean_answer_and_context(ex, docs_dir:Path):
    """Return cleaned example and list of issues fixed."""
    issues = []
    q = (ex.get('question') or '').strip()
    c = (ex.get('context') or '').strip()
    a = (ex.get('answer') or '').strip()

    # remove common 'Extracted: TIMESTAMP ---' noise from context
    c_clean = re.sub(r'Extracted:.*?---', '', c, flags=re.IGNORECASE | re.DOTALL)
    c_clean = re.sub(r'\s{2,}', ' ', c_clean).strip()

    # extract sources from answer or context
    refs = extract_sources(a)
    if not refs:
        refs = extract_sources(c_clean)

    # normalize refs to basenames and check existence
    clean_refs = []
    missing = []
    for r in refs:
        b = normalize_basename(r)
        # try to find file under docs with this basename
        found = None
        for p in docs_dir.rglob('*'):
            if p.name == b:
                found = p
                break
        if found:
            clean_refs.append(found.name)
        else:
            # if basename has weird long prefixes, try again with only last path segment
            clean_refs.append(b)
            # we won't mark missing as fatal; record
            missing.append(b)

    # ensure uniqueness and order
    seen = set()
    clean_refs = [x for x in clean_refs if not (x in seen or seen.add(x))]

    # ensure answer contains a Sources: line with bracketed basenames
    # remove existing Sources: lines from answer
    a_no_sources = re.sub(r'Source[s]?:.*', '', a, flags=re.IGNORECASE | re.DOTALL).strip()
    # collapse whitespace
    a_no_sources = re.sub(r'\s{2,}', ' ', a_no_sources)

    if clean_refs:
        sources_token = 'Sources: ' + ', '.join([f'[{os.path.basename(x)}]' for x in clean_refs])
        if a_no_sources.endswith('.'):
            new_answer = a_no_sources + ' ' + sources_token
        else:
            new_answer = a_no_sources + '. ' + sources_token
    else:
        # no refs found; keep original answer but ensure Sources token mentions unknown
        if 'source' not in a_no_sources.lower():
            new_answer = a_no_sources + ' Sources: [unknown]'
        else:
            new_answer = a_no_sources

    # final cleaning: strip extra whitespace
    c_final = c_clean.strip()
    a_final = new_answer.strip()

    cleaned = {
        'context': c_final,
        'question': q,
        'answer': a_final
    }
    if missing:
        issues.append('missing_files:' + ','.join(missing))
    return cleaned, issues


def check_example(ex, docs_dir:Path, min_context=50, min_answer=30):
    issues = []
    q = (ex.get('question') or '').strip()
    c = (ex.get('context') or '').strip()
    a = (ex.get('answer') or '').strip()
    if not q:
        issues.append('empty_question')
    elif len(q) < 8:
        issues.append('short_question')
    if not c:
        issues.append('empty_context')
    elif len(c) < min_context:
        issues.append('short_context')
    if not a:
        issues.append('empty_answer')
    elif len(a) < min_answer:
        issues.append('short_answer')
    if 'sources:' not in a.lower() and 'source:' not in a.lower():
        # allow sources in context too
        if 'sources:' not in c.lower() and 'source:' not in c.lower():
            issues.append('missing_sources_token')
    # try to detect referenced files
    refs = extract_sources(a) or extract_sources(c)
    missing = []
    for r in refs:
        candidate = docs_dir / r
        if not candidate.exists():
            found = False
            for p in docs_dir.rglob('*'):
                if p.name == r:
                    found = True
                    break
            if not found:
                missing.append(r)
    if missing:
        issues.append('missing_files:' + ','.join(missing))
    return {'question': q, 'context_len': len(c), 'answer_len': len(a), 'refs': refs, 'issues': issues}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default='docs/rag_training_examples_expanded.jsonl')
    p.add_argument('--sample', type=int, default=50)
    p.add_argument('--min_context', type=int, default=50)
    p.add_argument('--min_answer', type=int, default=30)
    p.add_argument('--out', default='docs/rag_examples_report.csv')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--clean', action='store_true', help='Produce a cleaned JSONL file')
    p.add_argument('--clean_out', default='docs/rag_training_examples_cleaned.jsonl')
    args = p.parse_args()

    docs_dir = Path('docs')
    examples = load_examples(args.file)
    n = len(examples)
    if n == 0:
        print('No examples found in', args.file)
        return

    # If cleaning requested, process all and write cleaned JSONL
    if args.clean:
        cleaned_all = []
        missing_count = 0
        for ex in examples:
            cleaned, issues = clean_answer_and_context(ex, docs_dir)
            if issues:
                missing_count += 1
            cleaned_all.append(cleaned)
        # write cleaned file
        outp = Path(args.clean_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open('w', encoding='utf-8') as fh:
            for e in cleaned_all:
                fh.write(json.dumps(e, ensure_ascii=False) + '\n')
        print(f'Wrote {len(cleaned_all)} cleaned examples to {outp}. {missing_count} had missing file references.')
        # continue to sampling/validation on cleaned set
        examples = cleaned_all
        n = len(examples)

    random.seed(args.seed)
    sample_size = min(args.sample, n)
    sampled = random.sample(examples, sample_size)

    rows = []
    seen = set()
    dup_count = 0
    for i, ex in enumerate(sampled):
        res = check_example(ex, docs_dir, min_context=args.min_context, min_answer=args.min_answer)
        key = (res['question'].lower(), ex.get('answer','').strip())
        is_dup = key in seen
        if is_dup:
            dup_count += 1
        seen.add(key)
        rows.append({
            'idx': i,
            'question': res['question'],
            'context_len': res['context_len'],
            'answer_len': res['answer_len'],
            'refs': ';'.join(res['refs']),
            'issues': ';'.join(res['issues']) if res['issues'] else '',
            'is_duplicate': is_dup,
        })

    # write CSV
    with open(args.out, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['idx','question','context_len','answer_len','refs','issues','is_duplicate'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary
    total_issues = sum(1 for r in rows if r['issues'])
    print(f'Sampled {sample_size} of {n} examples. Report written to {args.out}')
    print(f'Examples with issues: {total_issues}, duplicates in sample: {dup_count}')
    print('Top issues samples (first 10 rows with issues):')
    count = 0
    for r in rows:
        if r['issues']:
            print(f"- idx={r['idx']} issues={r['issues']} refs={r['refs']} q='{r['question'][:80]}'")
            count += 1
            if count >= 10:
                break

if __name__ == '__main__':
    main()
