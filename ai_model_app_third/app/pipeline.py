import os
import json
import numpy as np
import torch
import glob
from utils.preproc import prepare_data
from utils.fusion import multimodal_fusion_model
from utils.metrics import evaluate_model, plot_metrics
from models.cnn_backbones import load_cnn_model
from models.trained import train_model
from models.text_classifier import train_text_classifier


def train_text_model_from_embeddings(emb_path=None, labels_csv=None, out_model_path='docs/text_model.pth', epochs=20):
    """Train a text classifier from embeddings. Robustly locate and convert sparse TF-IDF + meta json files
    under the docs/ folder when available, then call the trainer on a dense .npz with arrays 'embeddings' and 'paths'.
    """
    docs_dir = os.path.join('docs')
    emb_path = emb_path or os.path.join(docs_dir, 'embeddings.npz')
    labels_csv = labels_csv or os.path.join(docs_dir, 'labels.csv')

    # Helper: try to read meta JSON near a candidate path
    def _read_meta(candidate):
        meta_candidates = [candidate + '.meta.json', os.path.splitext(candidate)[0] + '.meta.json']
        for m in meta_candidates:
            if os.path.exists(m):
                try:
                    with open(m, 'r', encoding='utf-8') as mf:
                        return json.load(mf)
                except Exception:
                    continue
        return None

    # 1) If a meta JSON exists in docs/ with many paths, prefer converting the sparse TF-IDF that generated it
    meta_files = [p for p in [
        os.path.join(docs_dir, 'embeddings.npz.meta.json'),
        os.path.join(docs_dir, 'embeddings_converted.npz.meta.json'),
        os.path.join(docs_dir, 'embeddings_dense.npz.meta.json')
    ] if os.path.exists(p)]

    converted_path = os.path.join(docs_dir, 'embeddings_converted.npz')

    if meta_files:
        # pick first meta and try to find a sparse .npz that matches
        meta = None
        for mf in meta_files:
            meta = _read_meta(os.path.splitext(mf)[0])
            if meta:
                break
        if meta and isinstance(meta.get('paths'), list) and len(meta['paths']) > 1:
            n_meta = len(meta['paths'])
            # look for candidate .npz files in docs/ that might be sparse
            candidates = glob.glob(os.path.join(docs_dir, 'embeddings*.npz'))
            # ensure emb_path candidate included
            if emb_path not in candidates and os.path.exists(emb_path):
                candidates.insert(0, emb_path)

            for cand in candidates:
                try:
                    # try load by scipy (sparse) first
                    import scipy.sparse as sp
                    mat = sp.load_npz(cand)
                    if hasattr(mat, 'shape') and mat.shape[0] == n_meta:
                        emb = mat.toarray()
                        paths = meta.get('paths', [])
                        np.savez_compressed(converted_path, embeddings=emb, paths=np.array(paths, dtype=object))
                        print(f"Auto-converted sparse {cand} -> {converted_path} (n={emb.shape[0]})")
                        use_path = converted_path
                        return train_text_classifier(emb_path=use_path, labels_csv_path=labels_csv, out_model_path=out_model_path, epochs=epochs)
                except Exception:
                    # not a sparse .npz or failed; try next
                    pass

    # 2) Try to directly interpret emb_path (numpy .npz with embeddings or object arrays)
    try:
        arr = np.load(emb_path, allow_pickle=True)
        files = getattr(arr, 'files', None)
        if files and 'embeddings' in files:
            embeddings = arr['embeddings']
            paths = arr['paths'].tolist() if 'paths' in files else []
            # if embeddings looks like object array of length>1, vstack
            embeddings = np.asarray(embeddings)
            if embeddings.dtype == object:
                try:
                    embeddings = np.vstack([np.asarray(x) for x in embeddings])
                except Exception:
                    if embeddings.size == 1 and isinstance(embeddings[0], (list, np.ndarray)):
                        embeddings = np.asarray(embeddings[0])
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            # Save a converted dense file to ensure trainer sees 'paths'
            np.savez_compressed(converted_path, embeddings=embeddings, paths=np.array(paths if paths else [], dtype=object))
            print(f"Interpreted numpy archive {emb_path} -> {converted_path} (n={embeddings.shape[0]})")
            return train_text_classifier(emb_path=converted_path, labels_csv_path=labels_csv, out_model_path=out_model_path, epochs=epochs)
    except Exception:
        pass

    # 3) Last resort: try scipy sparse load from emb_path even if meta not present
    try:
        import scipy.sparse as sp
        mat = sp.load_npz(emb_path)
        emb = mat.toarray()
        # try to read meta JSON for paths
        meta = _read_meta(emb_path)
        paths = meta.get('paths', []) if meta else []
        np.savez_compressed(converted_path, embeddings=emb, paths=np.array(paths if paths else [], dtype=object))
        print(f"Converted sparse {emb_path} -> {converted_path} (n={emb.shape[0]})")
        return train_text_classifier(emb_path=converted_path, labels_csv_path=labels_csv, out_model_path=out_model_path, epochs=epochs)
    except Exception:
        pass

    raise RuntimeError(f"Could not locate or convert embeddings for training from {emb_path} and docs/; check files and meta JSON.")

def define_steps(modality_choice, mri_types=None):
    data = prepare_data(modality_choice, mri_types)
    return data

def get_input_shape_from_sample(image):
    if isinstance(image, list):
        if len(image) == 0:
            raise ValueError("Le premier élément de data['images'] est une liste vide.")
        image = image[0]
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    return image.shape

def initialize_image_model(model_choice, data):
    if 'images' not in data or len(data['images']) == 0:
        raise ValueError("Aucune image n'a été trouvée dans les données.")
    print(f"🧪 data['images'] contient {len(data['images'])} éléments")
    input_shape = get_input_shape_from_sample(data['images'][0])
    print(f"📐 input_shape détectée : {input_shape}")
    return load_cnn_model(model_choice, input_shape)

def build_doc_embeddings(docs_dir='docs', output_path=None):
    """Create embeddings (or TF-IDF matrix) for all .txt/.md files in docs_dir.
    Saves results to output_path (defaults to docs/embeddings.npz) and returns a summary dict.
    Uses sentence-transformers when available, otherwise falls back to TF-IDF.
    Always writes a meta JSON file next to the embeddings containing 'paths' so the UI can read filenames.
    """
    import json
    if output_path is None:
        output_path = os.path.join(docs_dir, 'embeddings.npz')
    meta_path = output_path + '.meta.json'

    texts = []
    paths = []
    for root, _, files in os.walk(docs_dir):
        for f in files:
            if f.lower().endswith('.txt') or f.lower().endswith('.md'):
                full = os.path.join(root, f)
                try:
                    with open(full, 'r', encoding='utf-8') as fh:
                        txt = fh.read()
                        texts.append(txt)
                        paths.append(full)
                except Exception:
                    continue

    if len(texts) == 0:
        raise RuntimeError(f"No text documents found under {docs_dir}")

    summary = {'n_docs': len(texts), 'docs_dir': docs_dir}

    # Try sentence-transformers for dense embeddings
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb = model.encode(texts, convert_to_numpy=True)
        # Save dense embeddings and paths in a .npz
        np.savez_compressed(output_path, embeddings=emb, paths=np.array(paths, dtype=object))
        # Save meta JSON with paths
        with open(meta_path, 'w', encoding='utf-8') as mf:
            json.dump({'paths': paths, 'method': 'sentence-transformers'}, mf)
        summary['method'] = 'sentence-transformers'
        summary['output'] = output_path
        summary['meta'] = meta_path
        return summary
    except Exception:
        # continue to TF-IDF fallback
        pass

    # Fallback TF-IDF
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        vec = TfidfVectorizer(max_features=20000, stop_words='english')
        mat = vec.fit_transform(texts)
        # Save sparse matrix to output_path using scipy
        import scipy.sparse as sp
        sp.save_npz(output_path, mat)
        # Save meta JSON with paths and vocab info
        meta = {'paths': paths, 'method': 'tfidf', 'vocab': vec.get_feature_names_out().tolist()}
        with open(meta_path, 'w', encoding='utf-8') as mf:
            json.dump(meta, mf)
        summary['method'] = 'tfidf'
        summary['output'] = output_path
        summary['meta'] = meta_path
        return summary
    except Exception as e:
        raise RuntimeError(f"Failed to build document embeddings or TF-IDF: {e}")


# --- RAG support functions ---

def build_rag_index(embeddings_path, meta_path=None, index_path=None):
    """Build a nearest-neighbor index (FAISS if available, else sklearn) from a dense embeddings .npz or a scipy sparse .npz.
    Returns dict with index_path and n_docs.
    """
    if index_path is None:
        index_path = os.path.join(os.path.dirname(embeddings_path), 'rag_index')

    emb = None
    paths = []

    # Try numpy .npz with named arrays
    try:
        arr = np.load(embeddings_path, allow_pickle=True)
        files = getattr(arr, 'files', None)
        if files and 'embeddings' in files:
            emb = arr['embeddings']
            if 'paths' in files:
                p = arr['paths']
                # convert to python list
                try:
                    paths = p.tolist()
                except Exception:
                    paths = list(p)
    except Exception:
        emb = None

    # If not loaded, try scipy sparse .npz
    if emb is None:
        try:
            import scipy.sparse as sp
            mat = sp.load_npz(embeddings_path)
            emb = mat.toarray()
            # try to load meta JSON for paths
            if meta_path and os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as mf:
                        meta = json.load(mf)
                        paths = meta.get('paths', []) or []
                except Exception:
                    paths = []
        except Exception:
            emb = None

    if emb is None:
        raise RuntimeError(f"Could not read embeddings from {embeddings_path}")

    emb = np.asarray(emb)
    n_docs = emb.shape[0]

    # try faiss
    try:
        import faiss
        d = emb.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(emb.astype(np.float32))
        faiss.write_index(index, index_path + '.faiss')
        meta = {'index_type': 'faiss', 'index_file': index_path + '.faiss', 'n_docs': n_docs}
        with open(index_path + '.meta.json', 'w', encoding='utf-8') as mf:
            json.dump({'paths': paths, **meta}, mf)
        return {'index_path': index_path + '.faiss', 'n_docs': n_docs, 'paths': paths}
    except Exception:
        # fallback to sklearn nearest neighbors
        try:
            from sklearn.neighbors import NearestNeighbors
            import pickle
            nn = NearestNeighbors(n_neighbors=min(10, max(1, n_docs)), algorithm='auto').fit(emb)
            with open(index_path, 'wb') as fh:
                pickle.dump({'nn': nn, 'paths': paths}, fh)
            return {'index_path': index_path, 'n_docs': n_docs, 'paths': paths}
        except Exception as e:
            raise RuntimeError(f"Failed to build RAG index: {e}")


def prepare_rag_dataset_from_docs(docs_dir='docs', chunk_size=400):
    """Prepare simple RAG dataset by chunking documents into contexts.
    Returns list of dicts {'context': str, 'source': path}
    """
    contexts = []
    for root, _, files in os.walk(docs_dir):
        for f in files:
            if f.lower().endswith('.txt') or f.lower().endswith('.md'):
                full = os.path.join(root, f)
                try:
                    text = open(full, 'r', encoding='utf-8').read()
                    # split into paragraphs
                    paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
                    for p in paras:
                        # further chunk large paragraphs
                        for i in range(0, len(p), chunk_size):
                            ctx = p[i:i+chunk_size]
                            contexts.append({'context': ctx, 'source': full})
                except Exception:
                    continue
    return contexts


def train_rag_generator(contexts, out_dir='docs/rag_generator', epochs=1, model_name='t5-small', input_max_length=512, target_max_length=256):
    """Fine-tune a T5 generator robustly using transformers Seq2SeqTrainer.

    Uses DataCollatorForSeq2Seq to handle padding and label padding with -100 so loss ignores padded tokens.
    """
    try:
        from transformers import T5ForConditionalGeneration, T5TokenizerFast, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
        import datasets
    except Exception as e:
        raise RuntimeError(f"Transformers/datasets not available for generator training: {e}")

    if len(contexts) == 0:
        raise RuntimeError("No contexts provided for generator training")

    # Build source/target pairs (here we use autoencoding-style: context -> context)
    src_texts = [c['context'] for c in contexts]
    tgt_texts = src_texts

    ds = datasets.Dataset.from_dict({'src': src_texts, 'tgt': tgt_texts})

    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    def preprocess(batch):
        inputs = ["answer: " + s for s in batch['src']]
        model_inputs = tokenizer(inputs, max_length=input_max_length, truncation=True)
        labels = tokenizer(batch['tgt'], max_length=target_max_length, truncation=True)
        # replace pad token id in labels by -100 so it is ignored by loss
        label_ids = labels['input_ids']
        label_ids = [[(l if l != tokenizer.pad_token_id else -100) for l in seq] for seq in label_ids]
        model_inputs['labels'] = label_ids
        return model_inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=['src','tgt'])

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        save_total_limit=2,
        fp16=False,
        logging_steps=50,
        gradient_accumulation_steps=1,
        save_strategy='no'
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(out_dir)

    return {'model_dir': out_dir, 'n_examples': len(src_texts)}


def run_pipeline(modality_choice, model_choice, mri_types=None, llm_context=None, extra_inputs=None):
    """
    Run the training pipeline. Accepts optional llm_context (assistant suggestion text or dict)
    and extra_inputs (e.g., uploaded files). These are currently logged and returned but not
    used directly by the training routines. Backward compatible with previous signature.
    """
    print(f"Modèle choisi : {model_choice}")
    print(f"Modalités choisies : {modality_choice}")
    if mri_types:
        print(f"Types de MRI sélectionnés : {mri_types}")
    if llm_context:
        print(f"LLM context / suggestion: {llm_context}")
    if extra_inputs:
        print(f"Extra inputs provided: {list(extra_inputs.keys()) if isinstance(extra_inputs, dict) else type(extra_inputs)}")

    # Special case: docs-only training -> build embeddings and return
    if modality_choice in ('docs_only', 'docs_embeddings'):
        docs_dir = 'docs'
        print(f"Preparing docs-only training from {docs_dir}...")

        # Use existing prepare_data to load embeddings/labels if implemented
        data = define_steps(modality_choice, mri_types)
        if not data or data.get('type') != 'text':
            # fallback: build embeddings from raw docs then call build_doc_embeddings
            print("No precomputed text data returned by prepare_data(), building embeddings...")
            summary = build_doc_embeddings(docs_dir=docs_dir)
            # after building, point to the produced embeddings file
            emb_path = summary.get('output')
            labels_csv = os.path.join(docs_dir, 'labels.csv') if os.path.exists(os.path.join(docs_dir, 'labels.csv')) else None
            print(f"Embeddings built at {emb_path}, labels: {labels_csv}")
            res = train_text_model_from_embeddings(emb_path=emb_path, labels_csv=labels_csv, out_model_path=os.path.join(docs_dir, 'text_model.pth'), epochs=20)
            return {'docs_summary': summary, 'text_training': res}

        # If prepare_data returned text embeddings in memory, persist them to a file the trainer can read
        emb_tensor = data.get('embeddings')
        paths = data.get('paths', []) or []
        labels_tensor = data.get('labels', None)

        if emb_tensor is None:
            raise RuntimeError("prepare_data did not return embeddings for docs-only modality.")

        # Convert embeddings to numpy and save
        emb_np = emb_tensor.detach().cpu().numpy() if hasattr(emb_tensor, 'detach') else np.asarray(emb_tensor)
        converted_path = os.path.join(docs_dir, 'embeddings_for_training.npz')
        np.savez_compressed(converted_path, embeddings=emb_np, paths=np.array(paths, dtype=object))
        print(f"Saved embeddings for training: {converted_path} (n_samples={emb_np.shape[0]})")

        # If labels tensor provided, write a temporary labels CSV using basenames
        labels_csv_path = None
        if labels_tensor is not None:
            try:
                import pandas as _pd
                lbls = labels_tensor.detach().cpu().numpy() if hasattr(labels_tensor, 'detach') else np.asarray(labels_tensor)
                # Build rows using paths basenames when available, else index-based names
                rows = []
                for i in range(len(lbls)):
                    name = os.path.basename(paths[i]) if i < len(paths) and paths[i] else f'doc_{i}.txt'
                    rows.append({'filename': name, 'label': int(lbls[i])})
                labels_csv_path = os.path.join(docs_dir, 'labels_for_training.csv')
                _pd.DataFrame(rows)[['filename', 'label']].to_csv(labels_csv_path, index=False)
                print(f"Wrote temporary labels CSV: {labels_csv_path}")
            except Exception as e:
                print(f"Failed to write labels CSV: {e}")
                labels_csv_path = None

        # Train text classifier from the saved embeddings file
        try:
            text_res = train_text_model_from_embeddings(emb_path=converted_path, labels_csv=labels_csv_path, out_model_path=os.path.join(docs_dir, 'text_model.pth'), epochs=20)
            print("Text classifier training finished.")
        except Exception as e:
            raise RuntimeError(f"Text classifier training failed: {e}")

        # Evaluate the trained text model using the embeddings + labels
        try:
            # load embeddings we saved
            arr = np.load(converted_path, allow_pickle=True)
            emb = arr['embeddings']
            paths_loaded = arr['paths'].tolist() if 'paths' in getattr(arr, 'files', []) else []

            # Determine labels CSV to use
            eval_labels_csv = labels_csv_path or (os.path.join(docs_dir, 'labels.csv') if os.path.exists(os.path.join(docs_dir, 'labels.csv')) else None)
            labels_array = None
            if eval_labels_csv and os.path.exists(eval_labels_csv):
                import pandas as _pd
                df = _pd.read_csv(eval_labels_csv)
                if 'filename' in df.columns and 'label' in df.columns:
                    mapping = {str(r['filename']): r['label'] for _, r in df.iterrows()}
                    # Build labels list in order of paths_loaded when possible
                    if paths_loaded and len(paths_loaded) == emb.shape[0]:
                        labels_list = []
                        for p in paths_loaded:
                            b = os.path.basename(p)
                            labels_list.append(mapping.get(b))
                        if all([l is not None and str(l).strip() != '' for l in labels_list]):
                            uniques = sorted(list(set(labels_list)))
                            label2idx = {lab: idx for idx, lab in enumerate(uniques)}
                            labels_array = np.array([label2idx[l] for l in labels_list], dtype=int)
                    else:
                        # fallback: try based on filename order in df matching first N docs
                        basenames = df['filename'].astype(str).tolist()
                        if len(basenames) >= emb.shape[0]:
                            labels_list = basenames[:emb.shape[0]]
                            uniques = sorted(list(set(labels_list)))
                            label2idx = {lab: idx for idx, lab in enumerate(uniques)}
                            labels_array = np.array([label2idx[l] for l in labels_list], dtype=int)

            # If labels_array still None and prepare_data provided labels_tensor earlier use that
            if labels_array is None and labels_tensor is not None:
                labels_array = labels_tensor.detach().cpu().numpy() if hasattr(labels_tensor, 'detach') else np.asarray(labels_tensor)

            if labels_array is None:
                raise RuntimeError('No labels available for evaluation of text model.')

            # Build evaluation data dict
            eval_data = {'embeddings': emb, 'labels': labels_array, 'paths': paths_loaded}

            # Load trained model
            ckpt_path = text_res.get('model_path') if isinstance(text_res, dict) and text_res.get('model_path') else os.path.join(docs_dir, 'text_model.pth')
            if not os.path.exists(ckpt_path):
                raise RuntimeError(f"Trained model not found at {ckpt_path}")

            from models.text_classifier import SimpleTextClassifier
            ckpt = torch.load(ckpt_path, map_location='cpu')
            input_dim = ckpt.get('input_dim')
            n_classes = ckpt.get('n_classes')
            model_txt = SimpleTextClassifier(input_dim=input_dim, hidden_dim=128, n_classes=n_classes)
            model_txt.load_state_dict(ckpt['model_state_dict'])

            # Evaluate
            eval_results = evaluate_model(model_txt, eval_data)
            try:
                plot_metrics(text_res.get('history'), eval_results)
            except Exception:
                print('Plot metrics failed for text model; continuing.')

        except Exception as e:
            raise RuntimeError(f"Text model evaluation failed: {e}")

        # Return training + evaluation summary
        return {
            'docs_summary': {'n_samples': emb_np.shape[0], 'emb_path': converted_path, 'paths_count': len(paths)},
            'text_training': text_res,
            'text_evaluation': eval_results
        }

    # Special case: rag training — build embeddings, index and optionally train generator
    if modality_choice in ('rag', 'rag_train'):
        docs_dir = 'docs'
        # Build embeddings (dense or tfidf as needed)
        summary = build_doc_embeddings(docs_dir=docs_dir)
        emb_path = summary.get('output')
        # Build index
        idx_res = build_rag_index(emb_path, meta_path=summary.get('meta'), index_path=os.path.join(docs_dir,'rag_index'))
        print(f"RAG index built: {idx_res}")

        train_res = None
        # Prepare contexts for generator training
        contexts = prepare_rag_dataset_from_docs(docs_dir=docs_dir, chunk_size=400)
        if len(contexts) < 10:
            print(f"Warning: only {len(contexts)} contexts found; generator training may be ineffective.")
        # Attempt to train generator if transformers available
        try:
            train_res = train_rag_generator(contexts, out_dir=os.path.join(docs_dir,'rag_generator'), epochs=1)
            print(f"RAG generator trained: {train_res}")
        except Exception as e:
            print(f"RAG generator training skipped: {e}")

        return {'rag_index': idx_res, 'rag_train': train_res, 'docs_summary': summary}

