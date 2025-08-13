# Top-level imports and sys.path fix
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix for torch 2.5.1 + Streamlit compatibility issue
# Prevent Streamlit's file watcher from probing torch.classes
try:
    import torch
    if 'torch.classes' in sys.modules:
        import types
        dummy = types.ModuleType("torch.classes")
        dummy.__path__ = []  # Empty path list to satisfy watcher
        sys.modules['torch.classes'] = dummy
except Exception:
    pass

import streamlit as st
from app.pipeline import run_pipeline, train_text_model_from_embeddings
from app import llm
from app import rag
from utils.pdf_ingest import ingest_pdf_file
from utils.metrics import plot_metrics, evaluate_model
from pathlib import Path
import numpy as np
import pandas as pd
import threading
import time

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

                # Try generator-based RAG answer first (prefers trained generator)
                assistant_text = None
                # Prefer generator-based RAG answer first (uses trained generator when available)
                try:
                    rag_res = rag.answer_question(message, top_k=4, generator_dir=os.path.join('docs','rag_generator'))
                    assistant_text = rag_res.get('answer')
                except Exception:
                    assistant_text = None

                # If generator did not produce an answer, fall back to document retrieval + naive synthesis
                if not assistant_text:
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
                    # Append assistant reply to session history
                    st.session_state.chat_history.append({'role': 'assistant', 'text': assistant_text, 'images': []})

                    # Show simple RAG status (generator vs extractive) if available
                    try:
                        used_gen = bool(rag_res.get('used_generator')) if isinstance(rag_res, dict) else False
                        sources = rag_res.get('sources') or [] if isinstance(rag_res, dict) else []
                        status = "Generator (synthesized answer)" if used_gen else "Extractive fallback (snippets)"
                        st.caption(f"RAG status: {status}. Sources: {', '.join(sources) if sources else 'none'}")
                    except Exception:
                        # defensive: do not break UI if rag_res shape unexpected
                        pass

                    # Debug: show generator load/generation status (safe access)
                    try:
                        dbg = rag.get_generator_debug()
                        gen_loaded = dbg.get('loaded')
                        gen_device = dbg.get('device')
                        load_err = dbg.get('load_error')
                        gen_err = dbg.get('last_gen_error')
                        st.caption(f"Generator loaded={gen_loaded} device={gen_device} | load_error={'yes' if load_err else 'no'} gen_error={'yes' if gen_err else 'no'}")
                        if load_err:
                            trace = dbg.get('load_trace') or ''
                            st.text_area('Generator load error (trace):', value=(str(load_err) + '\n\n' + trace).strip(), height=200)
                        if gen_err:
                            gtrace = dbg.get('last_gen_trace') or ''
                            st.text_area('Generator last generation error (trace):', value=(str(gen_err) + '\n\n' + gtrace).strip(), height=200)
                    except Exception:
                        pass
                

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
            # fallback: keep the existing dataframe if the newer editor is not available
            edited = st.session_state.labels_df
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
                                # filter out empty/None labels before sorting to avoid type errors
                                filtered = [l for l in labels if l is not None and str(l).strip()!='']
                                if filtered and len(filtered) == len(labels):
                                    uniques = sorted(list({l for l in filtered}))
                                    label2idx = {lab:i for i,lab in enumerate(uniques)}
                                    labels_array = np.array([label2idx[l] for l in labels], dtype=int)
                        if labels_array is None:
                            st.warning('No labels available for evaluation (provide docs/labels.csv).')
                        else:
                            eval_data = {'embeddings': emb, 'labels': labels_array, 'paths': paths}
                            # load model
                            ckpt = res if isinstance(res, dict) and res.get('model_path') else None
                            model_path = ckpt.get('model_path') if ckpt else os.path.join('docs','text_model.pth')
                            if model_path and os.path.exists(model_path):
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
    st.write("---")
    st.subheader("RAG (Retriever-Augmented Generation) training")
    rag_train_gen = st.checkbox("Train generator model (T5) after building index", value=False)
    rag_epochs = st.number_input("Generator training epochs", min_value=1, max_value=50, value=1)
    rag_chunk = st.number_input("Context chunk size (chars)", min_value=100, max_value=2000, value=400)
    rag_overlap = st.number_input("Context chunk overlap (chars)", min_value=0, max_value=1000, value=50)

    # Background passage-indexing: start thread and update session_state with progress
    def _run_passage_build(chunk_size, overlap):
        key = 'passage_build'
        st.session_state.setdefault(key, {})
        st.session_state[key].update({'status': 'running', 'progress': 0, 'error': None, 'result': None})
        try:
            passages = rag.build_passage_corpus('docs', chunk_size=int(chunk_size), overlap=int(overlap))
            st.session_state[key]['progress'] = 25
            meta = rag.embed_passages_and_save(passages, out_path='docs/passages_embeddings.npz')
            st.session_state[key]['progress'] = 65
            idx_res = rag.build_passage_index('docs/passages_embeddings.npz', index_path='docs/passages_index')
            st.session_state[key]['progress'] = 100
            st.session_state[key]['status'] = 'done'
            st.session_state[key]['result'] = {'meta': meta, 'index': idx_res, 'n_passages': len(passages), 'sample': passages[:5]}
        except Exception as e:
            st.session_state[key]['status'] = 'error'
            st.session_state[key]['error'] = str(e)

    if 'passage_build' not in st.session_state:
        st.session_state['passage_build'] = {'status': 'idle', 'progress': 0}

    if st.button("Build passage-level index (chunk docs into passages)"):
        if st.session_state['passage_build'].get('status') == 'running':
            st.warning('Passage build already running in background')
        else:
            # start background thread
            t = threading.Thread(target=_run_passage_build, args=(rag_chunk, rag_overlap), daemon=True)
            t.start()
            st.success('Passage indexing started in background — refresh status below')

    # Status display
    pb = st.session_state.get('passage_build', {})
    st.write('Passage indexing status: ', pb.get('status'))
    try:
        prog = int(pb.get('progress', 0))
    except Exception:
        prog = 0
    st.progress(prog)
    if pb.get('status') == 'error':
        st.error(f"Passage build error: {pb.get('error')}")
    if pb.get('status') == 'done' and pb.get('result'):
        res = pb['result']
        st.success(f"Passage index ready — {res.get('n_passages')} passages. Index: {res.get('index')}")
        st.subheader('Sample passages')
        for i, p in enumerate(res.get('sample', [])):
            st.markdown(f"**[{i+1}]** {os.path.basename(p['source'])} (chunk {p['idx']})")
            st.write(p['text'][:600] + ('...' if len(p['text'])>600 else ''))
    if st.button('Refresh passage indexing status'):
        # Button click triggers a rerun by Streamlit naturally; set a timestamp to mark refresh
        st.session_state['passage_build_refresh_ts'] = time.time()
        st.success('Refreshed status')

    if st.button("Train RAG (build index + optional generator)"):
         with st.spinner("Building RAG index and optionally training generator..."):
             try:
                 # call pipeline rag training; it will build embeddings + index and attempt generator training
                 res = run_pipeline('rag_train' if rag_train_gen else 'rag', model_choice)
                 st.success("RAG pipeline finished")
                 st.json(res)
                 # If generator trained, show summary
                 if isinstance(res, dict) and res.get('rag_train'):
                     st.subheader('Generator training summary')
                     st.json(res.get('rag_train'))
             except Exception as e:
                 st.error(f"RAG training failed: {e}")
 
    st.caption('Notes: evaluation requires labels for the embeddings. Use the Labels tab to create docs/labels.csv.')

# --- Logs tab ---
with tabs[4]:
    st.subheader('Recent actions')
    st.markdown('- Last LLM suggestion shown in Chat tab (if any).')
    st.markdown('- Use Docs-only training to run end-to-end and automatically see evaluation results.')
