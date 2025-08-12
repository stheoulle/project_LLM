# Top-level imports and sys.path fix
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app.pipeline import run_pipeline, train_text_model_from_embeddings
from app import llm
from utils.pdf_ingest import ingest_pdf_file
from utils.metrics import plot_metrics, evaluate_model
from pathlib import Path
import numpy as np
import pandas as pd

st.set_page_config(page_title="AI Model App — Chat & Train", layout="wide")

# --- Header ---
st.title("🧠 AI Model — Chat, Ingest, Train & Eval")
st.markdown("A compact interface to ingest docs, create labels, train text classifiers and see evaluation automatically.")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'labels_df' not in st.session_state:
    st.session_state.labels_df = None
if 'last_eval' not in st.session_state:
    st.session_state.last_eval = None
if 'last_train' not in st.session_state:
    st.session_state.last_train = None

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    model_choice = st.selectbox("Base model (backbone)", ["resnet18", "radimagenet", "efficientnet"], index=0)
    manual_modality = st.selectbox("Manual modality", ["docs_only", "text", "images", "images+meta", "images+meta+reports"], index=0)
    st.markdown("---")
    st.subheader("Quick settings")
    default_emb = os.path.join('docs', 'embeddings_converted.npz')
    emb_path_input = st.text_input("Embeddings path", value=default_emb, key='sidebar_emb_path')
    labels_csv_input = st.text_input("Labels CSV path", value=os.path.join('docs', 'labels.csv'), key='sidebar_labels_path')
    epochs = st.number_input("Epochs", min_value=1, max_value=500, value=20, step=1)
    st.caption("Use Docs-only to build embeddings from docs/ and run end-to-end training+evaluation.")

# Main tabs
tabs = st.tabs(["Chat", "Ingest PDFs", "Labels", "Training & Evaluation", "Logs"])

# --- Chat tab ---
with tabs[0]:
    st.subheader("Chat — provide descriptions and images")
    col1, col2 = st.columns([3,1])
    with col1:
        # Chat input area (use form to avoid trying to reset widget state)
        with st.form("chat_form"):
            message = st.text_area("Message (medical report / description)", key="message_input", height=150)
            images = st.file_uploader("Attach images (optional)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="image_uploader")
            submitted = st.form_submit_button("Send")

        if submitted:
            if not message and not images:
                st.warning("Add text or images before sending")
            else:
                # append user message to chat history
                st.session_state.chat_history.append({'role': 'user', 'text': message or '', 'images': images or []})

                # Try to answer the question by retrieval over local docs (RAG-style extractive answer)
                assistant_text = None
                try:
                    retr = llm.get_indexer('docs')
                    # Prefer semantic query if available; indexer.query handles both embedding and TF-IDF
                    hits = retr.query(message, top_k=4)

                    if hits and len(hits) > 0:
                        parts = []
                        for h in hits:
                            # h may contain 'text' and 'path' and 'score'
                            snippet = ''
                            if isinstance(h, dict):
                                snippet = h.get('text','')
                                path = h.get('path') or h.get('path') if 'path' in h else h.get('path', None)
                            else:
                                snippet = str(h)
                                path = None

                            # Extract a short snippet: first 300 chars, or first 2 sentences
                            s = snippet.replace('\n', ' ').strip()
                            if not s:
                                continue
                            # take first two sentences if possible
                            sentences = [seg.strip() for seg in s.split('.') if seg.strip()]
                            short = (sentences[0] + ('. ' + sentences[1] + '.') if len(sentences) > 1 else sentences[0]) if sentences else s[:300]
                            parts.append({'source': os.path.basename(path) if path else 'doc', 'score': h.get('score', 0) if isinstance(h, dict) else None, 'snippet': short})

                        # Build assistant answer: list sources + combined synthesis
                        answer_lines = ["I found the following relevant documents and passages:"]
                        for p in parts:
                            score_str = f" (score={p['score']:.3f})" if p.get('score') is not None else ''
                            answer_lines.append(f"- {p['source']}{score_str}: {p['snippet']}")

                        # naive synthesis: combine snippets
                        synthesis = ' '.join([p['snippet'] for p in parts])
                        if len(synthesis) > 800:
                            synthesis = synthesis[:800].rsplit(' ',1)[0] + '...'

                        answer_lines.append('\nSynthesis: ' + synthesis)
                        assistant_text = '\n'.join(answer_lines)
                    else:
                        assistant_text = "No relevant documents found in the local index. Try ingesting PDFs or add documents to docs/."
                except Exception as e:
                    assistant_text = f"Retrieval failed: {e}"

                # Append assistant reply if produced
                if assistant_text:
                    st.session_state.chat_history.append({'role': 'assistant', 'text': assistant_text, 'images': []})

    with col2:
        st.markdown("**LLM Assistant**")
        if st.button("Analyze & Suggest"):
            with st.spinner("Analyzing..."):
                res = llm.analyze_chat_and_suggest(st.session_state.chat_history, None)
                st.session_state.last_llm = res
                st.success(f"Suggested: {res['modality']}")
                st.info(res['assistant_text'])

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

# --- Ingest PDFs tab ---
with tabs[1]:
    st.subheader("Ingest PDF files into docs/")
    st.markdown("Upload PDF files; they will be converted to text and saved under docs/ and indexed for retrieval.")
    uploaded = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
    if st.button("Ingest uploaded PDFs"):
        if not uploaded:
            st.warning("No PDFs uploaded")
        else:
            docs_dir = 'docs'
            upload_dir = os.path.join(docs_dir, 'uploads')
            Path(upload_dir).mkdir(parents=True, exist_ok=True)
            created, errors = [], []
            for f in uploaded:
                try:
                    save_path = os.path.join(upload_dir, f.name)
                    with open(save_path, 'wb') as out_f:
                        out_f.write(f.getbuffer())
                    out_txt = ingest_pdf_file(save_path, docs_dir=docs_dir, overwrite=True)
                    created.append(out_txt)
                except Exception as e:
                    errors.append(f"{f.name}: {e}")
            try:
                idx = llm.get_indexer('docs')
                idx.build_index()
            except Exception as e:
                st.error(f"Index rebuild failed: {e}")
            if created:
                st.success(f"Created {len(created)} docs")
                for c in created: st.write(c)
            if errors:
                st.error("Some failed:")
                for e in errors: st.write(e)

# --- Labels tab ---
with tabs[2]:
    st.subheader("Edit / create docs/labels.csv")
    emb_to_inspect = st.text_input("Embeddings file to inspect", value=emb_path_input)
    col_load1, col_load2 = st.columns(2)
    def load_from_meta(path):
        try:
            arr = np.load(path, allow_pickle=True)
            if hasattr(arr,'files') and 'paths' in arr.files:
                return [os.path.basename(p) for p in arr['paths']]
        except Exception:
            pass
        # fallback scan
        files = [f for f in os.listdir('docs') if f.lower().endswith('.txt') or f.lower().endswith('.md')]
        return sorted(files)
    if col_load1.button("Load filenames from embeddings"):
        basenames = load_from_meta(emb_to_inspect)
        df = pd.DataFrame({'filename': basenames, 'label': ['']*len(basenames)})
        st.session_state.labels_df = df
        st.success(f"Loaded {len(basenames)} entries")
    if col_load2.button("Load existing docs/labels.csv"):
        path = os.path.join('docs','labels.csv')
        if os.path.exists(path):
            st.session_state.labels_df = pd.read_csv(path)
            st.success('Loaded existing labels.csv')
        else:
            st.warning('No docs/labels.csv')
    if st.session_state.labels_df is not None:
        st.markdown("**Edit labels**")
        try:
            edited = st.data_editor(st.session_state.labels_df, num_rows='dynamic')
        except Exception:
            edited = st.experimental_data_editor(st.session_state.labels_df)
        st.session_state.labels_df = edited
        if st.button("Save labels to docs/labels.csv"):
            out = os.path.join('docs','labels.csv')
            st.session_state.labels_df.to_csv(out, index=False)
            st.success(f"Saved {out}")

# --- Training & Evaluation tab ---
with tabs[3]:
    st.subheader("Training & Evaluation")
    st.markdown("Choose a modality and start training. For docs-only use the 'Docs-only' action which will also run evaluation and show results.")
    col_a, col_b = st.columns(2)
    with col_a:
        choice = st.selectbox("Modality to train", [manual_modality, st.session_state.get('suggested_modality') or manual_modality], index=0)
        emb_path = st.text_input("Embeddings path (for manual training)", value=emb_path_input)
        labels_path = st.text_input("Labels CSV path (optional)", value=labels_csv_input)
        epochs_input = st.number_input("Epochs (manual training)", min_value=1, max_value=500, value=int(epochs))
    with col_b:
        if st.button("Start Docs-only (run pipeline)"):
            with st.spinner("Running docs-only pipeline (train + eval)..."):
                try:
                    res = run_pipeline('docs_only', model_choice)
                    st.success("Docs-only pipeline finished")
                    # show training + evaluation if present
                    if isinstance(res, dict) and 'text_evaluation' in res:
                        st.subheader('Evaluation report')
                        eval_res = res['text_evaluation']
                        try:
                            plot_metrics(res['text_training'].get('history') if isinstance(res.get('text_training'), dict) else None, eval_res, streamlit=True)
                        except Exception:
                            st.write(eval_res.get('report'))
                    else:
                        st.write(res)
                except Exception as e:
                    st.error(f"Docs-only training failed: {e}")
        if st.button("Start Manual training from embeddings"):
            with st.spinner("Converting/Training from embeddings..."):
                try:
                    # call conversion+trainer directly
                    res = train_text_model_from_embeddings(emb_path=emb_path, labels_csv=labels_path, epochs=int(epochs_input))
                    st.success("Manual training finished")
                    # attempt evaluation: load embeddings file and labels
                    conv = os.path.join('docs','embeddings_converted.npz')
                    if not os.path.exists(conv) and os.path.exists(emb_path):
                        conv = emb_path
                    if os.path.exists(conv):
                        arr = np.load(conv, allow_pickle=True)
                        emb = arr['embeddings']
                        paths = arr['paths'].tolist() if 'paths' in getattr(arr,'files',[]) else []
                        # build labels array
                        labels_array = None
                        if os.path.exists(labels_path):
                            df = pd.read_csv(labels_path)
                            mapping = {r['filename']: r['label'] for _, r in df.iterrows()}
                            if paths and len(paths)==emb.shape[0]:
                                labels = [mapping.get(os.path.basename(p)) for p in paths]
                                if all([l is not None and str(l).strip()!='' for l in labels]):
                                    uniques = sorted(list(set(labels)))
                                    label2idx = {lab:i for i,lab in enumerate(uniques)}
                                    labels_array = np.array([label2idx[l] for l in labels], dtype=int)
                        if labels_array is None:
                            st.warning('No labels available for evaluation (provide docs/labels.csv).')
                        else:
                            eval_data = {'embeddings': emb, 'labels': labels_array, 'paths': paths}
                            # load model
                            ckpt = res if isinstance(res, dict) and res.get('model_path') else None
                            model_path = ckpt.get('model_path') if ckpt else os.path.join('docs','text_model.pth')
                            if os.path.exists(model_path):
                                from models.text_classifier import SimpleTextClassifier
                                ck = __import__('torch').load(model_path, map_location='cpu')
                                m = SimpleTextClassifier(input_dim=ck['input_dim'], hidden_dim=128, n_classes=ck['n_classes'])
                                m.load_state_dict(ck['model_state_dict'])
                                eval_res = evaluate_model(m, eval_data)
                                plot_metrics(res.get('history') if isinstance(res, dict) else None, eval_res, streamlit=True)
                            else:
                                st.warning('Trained model not found for evaluation')
                except Exception as e:
                    st.error(f"Manual training failed: {e}")

# --- Logs tab ---
with tabs[4]:
    st.subheader('Recent actions')
    st.markdown('- Last LLM suggestion shown in Chat tab (if any).')
    st.markdown('- Use Docs-only training to run end-to-end and automatically see evaluation results.')

st.caption('Notes: evaluation requires labels for the embeddings. Use the Labels tab to create docs/labels.csv.')
