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

    manual_modality = st.selectbox(
        "🧬 Manual modality (use only if you want to override suggestion)",
        ["text", "images", "images+meta", "images+meta+reports"], index=0
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

    st.write("---")
    st.subheader("Upload / inputs summary")
    st.markdown(f"**Uploaded tabular:** {st.session_state.uploaded_tabular.name if st.session_state.uploaded_tabular else 'None'}")
    st.markdown(f"**Chat messages:** {len(st.session_state.chat_history)}")
    st.markdown(f"**Suggested modality:** {st.session_state.suggested_modality}")

    st.write("---")
    st.caption("Notes: \n- The suggestion is provided by a local retriever and heuristic; add docs/ (.txt/.md) to improve suggestions.\n- Ingested PDF text files are saved under docs/ and will be indexed for retrieval.")

sys.setrecursionlimit(10000)  # keep existing behavior if needed
