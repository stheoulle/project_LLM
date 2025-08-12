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

        # Optionally, we could compute evaluation metrics here using the training history
        return {'docs_summary': {'n_samples': emb_np.shape[0], 'emb_path': converted_path, 'paths_count': len(paths)}, 'text_training': text_res}

    # 🧩 Préparation des données
    data = define_steps(modality_choice, mri_types)

    # Phase 1 — entraînement du modèle image only
    image_model = initialize_image_model(model_choice, data)
    history = train_model(image_model, data)
    results = evaluate_model(image_model, data)
    try:
        plot_metrics(history, results)
    except Exception:
        print("plot_metrics failed or running headless; continuing.")
    print("Phase 1 terminée : modèle image entraîné et évalué.")
    print(f"Début de la phase 2 avec le modèle {model_choice} et les données préparées.")

    # Phase 2 — entraînement du modèle multimodal (avec le modèle image déjà entraîné)
    fusion_model = multimodal_fusion_model(image_model, data)
    print(f"Modèle de fusion multimodal créé avec {model_choice} comme backbone.")
    fusion_history = train_model(fusion_model, data)
    print("Phase 2 terminée : modèle multimodal entraîné.")
    fusion_results = evaluate_model(fusion_model, data)
    print("Évaluation du modèle multimodal terminée.")
    try:
        plot_metrics(fusion_history, fusion_results)
    except Exception:
        print("plot_metrics failed for fusion model; continuing.")
    print("Pipeline complet : modèle multimodal entraîné et évalué.")

    # Return a summary dictionary so callers (UI/CLI) can display or log info
    return {
        'phase1_history': history,
        'phase1_results': results,
        'fusion_history': fusion_history,
        'fusion_results': fusion_results,
        'llm_context': llm_context,
        'extra_inputs': extra_inputs
    }

