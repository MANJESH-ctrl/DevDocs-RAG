# frontend/app.py
import time
import json
import requests
import streamlit as st

API = st.secrets.get("API_URL", "http://localhost:8000")

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DEV-DOCS RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Custom CSS — ChatGPT-style dark theme
# ─────────────────────────────────────────
st.markdown("""
<style>
/* Overall background */
.stApp { background-color: #212121; color: #ececec; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #171717;
    border-right: 1px solid #2f2f2f;
}

/* Hide default streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }

/* Chat messages */
.user-bubble {
    background-color: #2f2f2f;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 8px 0 8px 15%;
    color: #ececec;
    font-size: 15px;
    line-height: 1.5;
}
.assistant-bubble {
    background-color: #1a1a1a;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 8px 15% 8px 0;
    color: #ececec;
    font-size: 15px;
    line-height: 1.5;
    border: 1px solid #2f2f2f;
}

/* Citation cards */
.citation-card {
    background-color: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-left: 3px solid #10a37f;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 12px;
    color: #aaaaaa;
}
.citation-title {
    color: #10a37f;
    font-weight: 600;
    font-size: 12px;
}

/* Upload area */
.upload-area {
    border: 2px dashed #3a3a3a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    background-color: #1a1a1a;
}

/* Status badge */
.status-ready {
    background-color: #0d3320;
    color: #10a37f;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.status-processing {
    background-color: #2a2200;
    color: #f5a623;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

/* Scrollable chat history in sidebar */
.history-item {
    background-color: #2a2a2a;
    border-radius: 8px;
    padding: 8px 10px;
    margin: 4px 0;
    font-size: 13px;
    color: #cccccc;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.history-item:hover { background-color: #333333; }

/* Input box */
[data-testid="stChatInput"] textarea {
    background-color: #2f2f2f !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 12px !important;
    color: #ececec !important;
}

/* Buttons */
.stButton > button {
    background-color: #2f2f2f;
    color: #ececec;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    width: 100%;
}
.stButton > button:hover {
    background-color: #10a37f;
    border-color: #10a37f;
    color: white;
}

/* Title */
.app-title {
    font-size: 22px;
    font-weight: 700;
    color: #ececec;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.app-subtitle {
    font-size: 12px;
    color: #666;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────
defaults = {
    "session_id":   None,
    "doc_status":   None,
    "doc_name":     None,
    "messages":     [],      
    "chat_history": [],       
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────
# Helper — stream response from API
# ─────────────────────────────────────────
def stream_answer(session_id: str, question: str):
    """Generator: yields (token, sources) where sources is None until last chunk."""
    sources = []
    buffer  = ""
    with requests.post(
        f"{API}/chat/{session_id}/stream",
        json={"question": question},
        stream=True,
        timeout=60
    ) as r:
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk:
                continue
            buffer += chunk
            # Extract sources metadata sent as first chunk
            if "__SOURCES__" in buffer:
                parts = buffer.split("__SOURCES__")
                if len(parts) >= 3:
                    try:
                        sources = json.loads(parts[1])
                    except Exception:
                        sources = []
                    buffer = parts[2]  # remainder is actual answer tokens
                continue
            yield buffer, sources
            buffer = ""
    # Final flush
    if buffer:
        yield buffer, sources

# ─────────────────────────────────────────
# Helper — render citations
# ─────────────────────────────────────────
def render_citations(sources: list):
    if not sources:
        return
    st.markdown("**📎 Sources**")
    for i, s in enumerate(sources, 1):
        section = s.get("section", "").strip()
        preview = s.get("preview", "").strip()
        label   = s.get("source", "unknown")
        st.markdown(f"""
        <div class="citation-card">
            <div class="citation-title">[{i}] {label}{' › ' + section if section else ''}</div>
            <div>{preview}...</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-title">🧠 DEV-DOCS RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Chat with your developer documentation</div>', unsafe_allow_html=True)
    st.divider()

    # ── Upload ──
    st.markdown("#### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF", type="pdf", label_visibility="collapsed"
    )

    if uploaded_file:
        if st.button("🚀 Upload & Process", use_container_width=True):
            with st.spinner("Uploading..."):
                resp = requests.post(
                    f"{API}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.session_id = data["session_id"]
                st.session_state.doc_status = "processing"
                st.session_state.doc_name   = uploaded_file.name
                st.session_state.messages   = []
                st.rerun()
            else:
                st.error(f"Upload failed: {resp.text}")

    st.divider()

    # ── Current doc status ──
    st.markdown("#### 📋 Current Document")
    if st.session_state.doc_name:
        st.markdown(f"**📁 {st.session_state.doc_name}**")
        if st.session_state.doc_status == "ready":
            st.markdown('<span class="status-ready">● Ready</span>', unsafe_allow_html=True)
        elif st.session_state.doc_status == "processing":
            st.markdown('<span class="status-processing">⟳ Processing</span>', unsafe_allow_html=True)
        elif st.session_state.doc_status == "failed":
            st.error("❌ Processing failed")
        if st.session_state.session_id:
            st.caption(f"ID: `{st.session_state.session_id[:12]}...`")
    else:
        st.caption("No document loaded yet.")

    st.divider()

    # ── Chat history ──
    st.markdown("#### 🕘 Chat History")
    if not st.session_state.messages:
        st.caption("No messages yet.")
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                preview = msg["content"][:45] + "..." if len(msg["content"]) > 45 else msg["content"]
                st.markdown(f'<div class="history-item">💬 {preview}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Action buttons ──
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("📄 New Doc", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

    # ── Download chat ──
    if st.session_state.messages:
        chat_export = "\n\n".join(
            f"{'You' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in st.session_state.messages
        )
        st.download_button(
            label="⬇️ Download Chat",
            data=chat_export,
            file_name=f"chat_{st.session_state.doc_name or 'export'}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ─────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────

# ── Processing spinner (polls until ready) ──
if st.session_state.doc_status == "processing":
    st.markdown("### ⚙️ Processing your document...")
    progress_bar = st.progress(0)
    status_text  = st.empty()
    elapsed      = 0
    while True:
        resp   = requests.get(f"{API}/status/{st.session_state.session_id}")
        result = resp.json()
        status = result.get("status")

        if status == "ready":
            st.session_state.doc_status = "ready"
            progress_bar.progress(100)
            status_text.success("✅ Ready to chat!")
            time.sleep(0.8)
            st.rerun()
            break
        elif status == "failed":
            st.session_state.doc_status = "failed"
            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            break
        else:
            elapsed += 3
            # Fake progress to give user feedback
            fake_progress = min(int((elapsed / 30) * 90), 90)
            progress_bar.progress(fake_progress)
            status_text.info(f"⏳ Still processing... ({elapsed}s elapsed)")
            time.sleep(3)

# ── No doc loaded ──
elif st.session_state.doc_status is None:
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px;'>
        <div style='font-size:64px'>🧠</div>
        <h2 style='color:#ececec; margin-top:16px'>DEV-DOCS RAG</h2>
        <p style='color:#666; font-size:16px'>Upload a PDF from the sidebar to start chatting</p>
        <p style='color:#444; font-size:13px'>Powered by Pinecone · Groq · FastEmbed</p>
    </div>
    """, unsafe_allow_html=True)

# ── Failed ──
elif st.session_state.doc_status == "failed":
    st.error("Document processing failed. Upload a new document from the sidebar.")

# ── Ready — show chat ──
elif st.session_state.doc_status == "ready":

    # Render existing messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
            if msg.get("sources"):
                render_citations(msg["sources"])

    # Chat input
    if prompt := st.chat_input("Ask anything about your document..."):
        # Show user message immediately
        st.markdown(
            f'<div class="user-bubble">{prompt}</div>',
            unsafe_allow_html=True
        )
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

        # Stream assistant response
        answer_placeholder = st.empty()
        full_answer = ""
        final_sources = []

        with st.spinner(""):
            for token, sources in stream_answer(st.session_state.session_id, prompt):
                full_answer += token
                final_sources = sources if sources else final_sources
                answer_placeholder.markdown(
                    f'<div class="assistant-bubble">{full_answer}▌</div>',
                    unsafe_allow_html=True
                )

        # Final render without cursor
        answer_placeholder.markdown(
            f'<div class="assistant-bubble">{full_answer}</div>',
            unsafe_allow_html=True
        )
        render_citations(final_sources)

        # Save to session
        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_answer,
            "sources": final_sources
        })
