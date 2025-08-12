# Ensure workspace root is on sys.path so imports like `app` and `utils` resolve when running Streamlit from the app/ folder
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app.pipeline import run_pipeline
from app import llm
from utils.pdf_ingest import ingest_pdf_file
from pathlib import Path

st.set_page_config(page_title="AI Model App — Chat & Train", layout="wide")

st.title("🧠 AI Model — Chat, Analyze & Train")
st.markdown(
    "Utilisez la zone de chat pour fournir du texte médical et/ou des images."
    " Cliquez sur 'Analyze & Suggest' pour que l'interface propose une configuration de training adaptée, puis lancez l'entraînement."
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {role: 'user'|'assistant', 'text': str, 'images': [UploadedFile,...]}

if 'uploaded_tabular' not in st.session_state:
    st.session_state.uploaded_tabular = None

if 'suggested_modality' not in st.session_state:
    st.session_state.suggested_modality = None

if 'assistant_message' not in st.session_state:
    st.session_state.assistant_message = None

# Sidebar: manual training controls
with st.sidebar:
    st.header("Training Controls")
    model_choice = st.selectbox("🎯 Base model", ["resnet18", "radimagenet", "efficientnet"], index=0)

    # Add 'docs_only' option to allow training on text corpus under docs/
    manual_modality = st.selectbox(
        "🧬 Manual modality (use only if you want to override suggestion)",
        ["docs_only", "text", "images", "images+meta", "images+meta+reports"], index=0
    )

    st.write("---")
    st.subheader("Upload auxiliary data")
    uploaded_tabular = st.file_uploader("Upload tabular CSV (optional)", type=["csv"], key="tabular_uploader")
    if uploaded_tabular is not None:
        st.session_state.uploaded_tabular = uploaded_tabular
        st.success(f"Tabular file uploaded: {uploaded_tabular.name}")

    st.write("---")
    st.caption("After analysis you can start training with suggested or manual modality.")

# Main layout: chat on left, controls on right
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat — provide descriptions, findings or images")

    # Chat input area
    message = st.text_area("Message (medical report / description)", key="message_input", height=120)
    images = st.file_uploader("Attach images (optional)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="image_uploader")

    send_col1, send_col2, send_col3 = st.columns([1,1,1])
    with send_col1:
        if st.button("Send message"):
            if not message and not images:
                st.warning("Ajoutez un message ou une image avant d'envoyer.")
            else:
                st.session_state.chat_history.append({
                    'role': 'user',
                    'text': message if message else "",
                    'images': images if images else []
                })
                # clear input
                st.session_state.message_input = ""
                st.session_state.image_uploader = None
                st.experimental_rerun()

    with send_col2:
        if st.button("Analyze & Suggest (LLM)"):
            with st.spinner("Contacting LLM (or using fallback)..."):
                result = llm.analyze_chat_and_suggest(st.session_state.chat_history, st.session_state.uploaded_tabular)
                st.session_state.suggested_modality = result['modality']
                st.session_state.assistant_message = result['assistant_text']
                st.success(f"Suggested modality: {st.session_state.suggested_modality}")

    with send_col3:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.session_state.suggested_modality = None
            st.session_state.assistant_message = None
            st.experimental_rerun()

    st.write("---")

    # Display chat history
    for i, m in enumerate(reversed(st.session_state.chat_history)):
        role = m.get('role', 'user')
        text = m.get('text', '')
        imgs = m.get('images', [])
        if role == 'user':
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")
        if imgs:
            cols = st.columns(min(3, len(imgs)))
            for idx, f in enumerate(imgs):
                try:
                    cols[idx % 3].image(f, caption=getattr(f, 'name', 'image'), use_column_width=True)
                except Exception:
                    cols[idx % 3].write("[image]")

    if st.session_state.assistant_message:
        st.write("---")
        st.subheader("LLM Assistant suggestion")
        st.info(st.session_state.assistant_message)

    # PDF ingestion UI section
    st.write("---")
    st.subheader("Ingest PDFs into local docs (for LLM retrieval)")
    st.markdown("Upload DSM-5 or other medical PDFs; they will be converted to text and saved under the project's docs/ directory.")

    pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
    ingest_col1, ingest_col2 = st.columns([1,1])
    docs_dir = "docs"
    upload_dir = os.path.join(docs_dir, "uploads")
    Path(upload_dir).mkdir(parents=True, exist_ok=True)

    with ingest_col1:
        if st.button("Ingest uploaded PDFs"):
            if not pdfs:
                st.warning("Aucun PDF téléchargé.")
            else:
                created = []
                errors = []
                for f in pdfs:
                    try:
                        save_path = os.path.join(upload_dir, f.name)
                        # write uploaded file to disk
                        with open(save_path, "wb") as out_f:
                            out_f.write(f.getbuffer())
                        # ingest saved pdf into docs/
                        out_txt = ingest_pdf_file(save_path, docs_dir=docs_dir, overwrite=True)
                        created.append(out_txt)
                    except Exception as e:
                        errors.append(f"{f.name}: {e}")

                # Rebuild local index
                try:
                    idx = llm.get_indexer(docs_dir)
                    idx.build_index()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

                if created:
                    st.success(f"Created {len(created)} text docs under {docs_dir}")
                    for c in created:
                        st.write(c)
                if errors:
                    st.error("Some files failed to ingest:")
                    for e in errors:
                        st.write(e)

    with ingest_col2:
        if st.button("Ingest PDFs from folder on server"):
            # Allows ingesting PDFs from a server-side folder, for advanced users
            folder = st.text_input("Server folder path", value="/path/to/pdfs")
            if not folder or not os.path.exists(folder):
                st.warning("Please provide an existing folder path on the server.")
            else:
                try:
                    from utils.pdf_ingest import ingest_folder
                    created = ingest_folder(folder, docs_dir=docs_dir, recursive=True, overwrite=False)
                    idx = llm.get_indexer(docs_dir)
                    idx.build_index()
                    st.success(f"Ingested {len(created)} files from {folder}")
                except Exception as e:
                    st.error(f"Folder ingestion failed: {e}")

with col2:
    st.subheader("Suggested / Manual Training")

    if st.session_state.suggested_modality:
        st.info(f"Suggested modality based on LLM: {st.session_state.suggested_modality}")

    start_col1, start_col2 = st.columns(2)
    def _collect_images_from_chat():
        imgs = []
        for m in st.session_state.chat_history:
            for f in m.get('images', []) or []:
                imgs.append(f)
        return imgs

    with start_col1:
        if st.button("Start Suggested Training"):
            modality_to_use = st.session_state.suggested_modality or manual_modality
            extra_inputs = {
                'tabular': st.session_state.uploaded_tabular,
                'images': _collect_images_from_chat()
            }
            with st.spinner(f"Starting training with {modality_to_use} and backbone {model_choice}..."):
                try:
                    run_pipeline(modality_to_use, model_choice, llm_context=st.session_state.assistant_message, extra_inputs=extra_inputs)
                    st.success("Training finished. Check console logs and metrics outputs.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    with start_col2:
        if st.button("Start Manual Training"):
            modality_to_use = manual_modality
            extra_inputs = {
                'tabular': st.session_state.uploaded_tabular,
                'images': _collect_images_from_chat()
            }
            with st.spinner(f"Starting manual training with {modality_to_use} and backbone {model_choice}..."):
                try:
                    run_pipeline(modality_to_use, model_choice, llm_context=st.session_state.assistant_message, extra_inputs=extra_inputs)
                    st.success("Manual training finished. Check console logs and metrics outputs.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    # New: dedicated Docs-only training button
    st.write("---")
    st.subheader("Docs-only training")
    st.markdown("Run training steps that operate only on the .txt/.md files under the project's `docs/` directory (build embeddings / TF-IDF).")
    if st.button("Start Docs-only Training"):
        with st.spinner("Building embeddings from docs/ ..."):
            try:
                res = run_pipeline('docs_only', model_choice)
                # Expecting a summary dict under 'docs_summary'
                docs_summary = res.get('docs_summary') if isinstance(res, dict) else None
                if docs_summary:
                    st.success(f"Docs-only completed: {docs_summary.get('n_docs')} docs processed using {docs_summary.get('method')}")
                    st.write(docs_summary)
                else:
                    st.info("Docs-only finished (no summary returned). Check logs.")
            except Exception as e:
                st.error(f"Docs-only training failed: {e}")

    st.write("---")
    st.subheader("Train text classifier from docs embeddings")
    st.markdown("If you have generated `docs/embeddings.npz`, you can train a simple classifier using `docs/labels.csv` (filename,label). If no labels, KMeans pseudo-labeling will be used if scikit-learn is installed.")

    emb_path_input = st.text_input("Embeddings path", value=os.path.join('docs', 'embeddings.npz'))
    labels_csv_input = st.text_input("Labels CSV path (optional)", value=os.path.join('docs', 'labels.csv'))
    epochs = st.number_input("Epochs", min_value=1, max_value=500, value=20)

    if st.button("Start text classifier training"):
        with st.spinner("Training text classifier from embeddings..."):
            try:
                from app.pipeline import train_text_model_from_embeddings
                res = train_text_model_from_embeddings(emb_path=emb_path_input, labels_csv=labels_csv_input, out_model_path=os.path.join('docs','text_model.pth'), epochs=int(epochs))
                st.success(f"Text model trained: {res.get('model_path')} — classes: {res.get('n_classes')} — samples: {res.get('n_samples')}")
                st.write(res.get('history'))
            except Exception as e:
                st.error(f"Text classifier training failed: {e}")

    # New: Labels CSV editor for docs embeddings
    st.write("---")
    st.subheader("Edit / Create labels CSV for docs embeddings")
    st.markdown("Use this tool to create or edit `docs/labels.csv` so the text classifier can be trained. Filenames must match the basenames stored in the embeddings file.")

    emb_path_for_labels = st.text_input("Embeddings file to inspect", value=os.path.join('docs', 'embeddings.npz'), key='labels_emb_path')
    load_col1, load_col2 = st.columns([1,1])

    def _load_paths_from_embeddings(path):
        import json, glob
        # If exact meta JSON exists, read it
        meta_candidates = []
        if os.path.exists(path):
            # if exact file is provided and there is a sibling meta json
            meta_candidates.append(path + '.meta.json')
        # if a meta JSON was provided directly
        if path.lower().endswith('.meta.json') and os.path.exists(path):
            meta_candidates.insert(0, path)

        # look for any matching meta json in same directory
        base_dir = os.path.dirname(path) if os.path.dirname(path) else 'docs'
        for p in glob.glob(os.path.join(base_dir, '*meta*.json')):
            meta_candidates.append(p)

        # try candidates
        for m in meta_candidates:
            try:
                if os.path.exists(m):
                    with open(m, 'r', encoding='utf-8') as mf:
                        meta = json.load(mf)
                        if 'paths' in meta and isinstance(meta['paths'], list) and len(meta['paths']) > 0:
                            return [os.path.basename(p) for p in meta['paths']]
            except Exception:
                continue

        # If no meta found, try to load numpy .npz that contains 'paths'
        try:
            import numpy as _np
            if os.path.exists(path):
                arr = _np.load(path, allow_pickle=True)
                if hasattr(arr, 'files') and 'paths' in arr.files:
                    return [os.path.basename(p) for p in arr['paths']]
        except Exception:
            pass

        # Fallback: scan docs directory for .txt/.md files
        try:
            files = []
            for root, _, fnames in os.walk('docs'):
                for fn in fnames:
                    if fn.lower().endswith('.txt') or fn.lower().endswith('.md'):
                        files.append(fn)
            return sorted(list(set(files)))
        except Exception:
            return []

    if load_col1.button("Load filenames from embeddings"):
        with st.spinner("Reading embeddings metadata ..."):
            basenames = _load_paths_from_embeddings(emb_path_for_labels)
            if not basenames:
                st.warning("No filenames found in embeddings or meta. Make sure embeddings were saved with 'paths' or meta JSON exists.")
            else:
                import pandas as _pd
                df = _pd.DataFrame({'filename': basenames, 'label': [''] * len(basenames)})
                st.session_state.labels_df = df
                st.success(f"Loaded {len(basenames)} filenames from embeddings")

    if load_col2.button("Load existing docs/labels.csv"):
        labels_csv_path = os.path.join('docs', 'labels.csv')
        if os.path.exists(labels_csv_path):
            import pandas as _pd
            try:
                df = _pd.read_csv(labels_csv_path)
                # ensure columns
                if 'filename' not in df.columns or 'label' not in df.columns:
                    st.error('Existing CSV must contain columns: filename,label')
                else:
                    st.session_state.labels_df = df
                    st.success(f"Loaded {labels_csv_path}")
            except Exception as e:
                st.error(f"Failed to load CSV: {e}")
        else:
            st.warning(f"No labels CSV found at {labels_csv_path}")

    # Show editable table if loaded
    if 'labels_df' in st.session_state and isinstance(st.session_state.labels_df, (list, tuple)) is False:
        df = st.session_state.labels_df
        st.markdown("**Edit labels below (filename, label)**")
        # Prefer newer data editor if available
        try:
            edited = st.data_editor(df, num_rows='dynamic')
        except Exception:
            try:
                edited = st.experimental_data_editor(df)
            except Exception:
                st.dataframe(df)
                edited = df

        # Save edited back to session
        st.session_state.labels_df = edited

        save_col1, save_col2 = st.columns([1,1])
        labels_csv_path = os.path.join('docs', 'labels.csv')
        with save_col1:
            if st.button("Save labels to docs/labels.csv"):
                try:
                    import pandas as _pd
                    os.makedirs(os.path.dirname(labels_csv_path), exist_ok=True)
                    _pd.DataFrame(st.session_state.labels_df).to_csv(labels_csv_path, index=False)
                    st.success(f"Saved labels to {labels_csv_path}")
                except Exception as e:
                    st.error(f"Failed to save labels CSV: {e}")
        with save_col2:
            if st.button("Clear labels table"):
                del st.session_state['labels_df']
                st.experimental_rerun()

    st.write("---")
    st.subheader("Upload / inputs summary")
    st.markdown(f"**Uploaded tabular:** {st.session_state.uploaded_tabular.name if st.session_state.uploaded_tabular else 'None'}")
    st.markdown(f"**Chat messages:** {len(st.session_state.chat_history)}")
    st.markdown(f"**Suggested modality:** {st.session_state.suggested_modality}")

    st.write("---")
    st.caption("Notes: \n- The suggestion is provided by a local retriever and heuristic; add docs/ (.txt/.md) to improve suggestions.\n- Ingested PDF text files are saved under docs/ and will be indexed for retrieval.")

sys.setrecursionlimit(10000)  # keep existing behavior if needed
