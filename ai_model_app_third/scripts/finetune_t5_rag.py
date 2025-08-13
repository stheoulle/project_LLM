#!/usr/bin/env python3
"""Fine-tune a T5 seq2seq model on JSONL training examples (context, question, answer).
Saves the fine-tuned model to the specified output directory (default: docs/rag_generator).

Example:
  python scripts/finetune_t5_rag.py --config scripts/train_config.json

The script expects the cleaned JSONL at docs/rag_training_examples_cleaned.jsonl by default.
"""
import argparse
import os
import json
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

PROMPT = (
    "You are an assistant that answers health questions in plain language for patients. "
    "Use ONLY the information in the contexts below. If the contexts do not contain an answer, say you do not know and advise consulting a healthcare professional. "
    "Answer concisely (1-3 short paragraphs) and list sources at the end in the format: Sources: [file1], [file2].\n\n"
    "CONTEXTS:\n{contexts}\n\nQUESTION: {question}\n\nAnswer:"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='scripts/train_config.json', help='Path to training config JSON')
    return p.parse_args()


def load_config(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_file = cfg.get('data_file', 'docs/rag_training_examples_cleaned.jsonl')
    model_name = cfg.get('model_name', 't5-small')
    output_dir = cfg.get('output_dir', 'docs/rag_generator')
    epochs = cfg.get('epochs', 3)
    per_device_train_batch_size = cfg.get('per_device_train_batch_size', 8)
    learning_rate = cfg.get('learning_rate', 3e-5)
    weight_decay = cfg.get('weight_decay', 0.01)
    max_input_length = cfg.get('max_input_length', 512)
    max_target_length = cfg.get('max_target_length', 150)
    fp16 = cfg.get('fp16', True)
    save_steps = cfg.get('save_steps', None)
    eval_steps = cfg.get('eval_steps', None)

    data_file = str(Path(data_file).expanduser())
    assert os.path.exists(data_file), f"Data file not found: {data_file}"

    print('Loading dataset from', data_file)
    ds = load_dataset('json', data_files=data_file, split='train')

    # Build input strings
    def build_input(example):
        contexts = example.get('context', '')
        question = example.get('question', '')
        return PROMPT.format(contexts=contexts, question=question)

    # Initialize tokenizer and model
    print('Loading tokenizer and model:', model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Preprocess
    def preprocess(batch):
        inputs = [build_input(x) for x in batch['context'].__class__ and batch['context']] if False else [PROMPT.format(contexts=c, question=q) for c, q in zip(batch['context'], batch['question'])]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding='max_length')
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch['answer'], max_length=max_target_length, truncation=True, padding='max_length')
        model_inputs['labels'] = labels['input_ids']
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        model_inputs['labels'] = [[(lab if lab != tokenizer.pad_token_id else -100) for lab in labs] for labs in model_inputs['labels']]
        return model_inputs

    print('Tokenizing dataset...')
    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training args (avoid passing evaluation_strategy/eval_steps for compatibility with older transformers)
    save_strategy = 'epoch' if save_steps is None else 'steps'
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        predict_with_generate=True,
        logging_strategy='steps',
        logging_steps=cfg.get('logging_steps', 100),
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=2,
        fp16=fp16 and tokenizer.model_max_length <= 1024,
        load_best_model_at_end=False,
        remove_unused_columns=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print('Starting training...')
    trainer.train()

    print('Saving model to', output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print('Done.')


if __name__ == '__main__':
    main()
