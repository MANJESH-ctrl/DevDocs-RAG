import time
import json
import requests
import streamlit as st

try:
    API = st.secrets["API_URL"]
except Exception:
    API = "http://localhost:8000"

st.set_page_config(
    page_title="DEV-DOCS RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Session state
# ─────────────────────────────────────────
defaults = {
    "session_id":    None,
    "doc_status":    None,
    "doc_name":      None,
    "messages":      [],
    "chat_history":  [],
    "sidebar_open":  True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────
# CSS — hide native collapse, style everything
# ─────────────────────────────────────────
sidebar_display = "flex" if st.session_state.sidebar_open else "none"

st.markdown(f"""
<style>
.stApp {{ background-color: #212121; color: #ececec; }}

[data-testid="stSidebar"] {{
    background-color: #171717;
    border-right: 1px solid #2f2f2f;
    display: {sidebar_display} !important;
}}

/* Hide Streamlit's native collapse button */
[data-testid="collapsedControl"] {{ display: none !important; }}
button[kind="header"] {{ display: none !important; }}

#MainMenu, footer, header {{ visibility: hidden; }}

.user-bubble {{
    background-color: #2f2f2f;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 8px 0 8px 15%;
    color: #ececec;
    font-size: 15px;
    line-height: 1.5;
}}
.assistant-bubble {{
    background-color: #1a1a1a;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 8px 15% 8px 0;
    color: #ececec;
    font-size: 15px;
    line-height: 1.5;
    border: 1px solid #2f2f2f;
}}
.citation-card {{
    background-color: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-left: 3px solid #10a37f;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 12px;
    color: #aaaaaa;
}}
.citation-title {{
    color: #10a37f;
    font-weight: 600;
    font-size: 12px;
}}
.status-ready {{
    background-color: #0d3320;
    color: #10a37f;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}}
.status-processing {{
    background-color: #2a2200;
    color: #f5a623;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}}
.history-item {{
    background-color: #2a2a2a;
    border-radius: 8px;
    padding: 8px 10px;
    margin: 4px 0;
    font-size: 13px;
    color: #cccccc;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
[data-testid="stChatInput"] textarea {{
    background-color: #2f2f2f !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 12px !important;
    color: #ececec !important;
}}
.stButton > button {{
    background-color: #2f2f2f;
    color: #ececec;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    width: 100%;
}}
.stButton > button:hover {{
    background-color: #10a37f;
    border-color: #10a37f;
    color: white;
}}
.toggle-btn > button {{
    background-color: #2f2f2f !important;
    color: #ececec !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 8px !important;
    width: auto !important;
    padding: 4px 12px !important;
    font-size: 18px !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Toggle button — always visible top-left
# ─────────────────────────────────────────
toggle_col, _ = st.columns([1, 11])
with toggle_col:
    with st.container():
        st.markdown('<div class="toggle-btn">', unsafe_allow_html=True)
        if st.button("☰", key="sidebar_toggle"):
            st.session_state.sidebar_open = not st.session_state.sidebar_open
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def stream_answer(session_id: str, question: str):
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
            if "__SOURCES__" in buffer:
                parts = buffer.split("__SOURCES__")
                if len(parts) >= 3:
                    try:
                        sources = json.loads(parts[1])
                    except Exception:
                        sources = []
                    buffer = parts[2]
                continue
            yield buffer, sources
            buffer = ""
    if buffer:
        yield buffer, sources


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
    st.markdown("### 🧠 DEV-DOCS RAG")
    st.caption("Chat with your developer documentation")
    st.divider()

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

    st.markdown("#### 🕘 Chat History")
    if not st.session_state.messages:
        st.caption("No messages yet.")
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                preview = msg["content"][:45] + "..." if len(msg["content"]) > 45 else msg["content"]
                st.markdown(f'<div class="history-item">💬 {preview}</div>', unsafe_allow_html=True)

    st.divider()

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
# MAIN AREA
# ─────────────────────────────────────────
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
            fake_progress = min(int((elapsed / 30) * 90), 90)
            progress_bar.progress(fake_progress)
            status_text.info(f"⏳ Still processing... ({elapsed}s elapsed)")
            time.sleep(3)

elif st.session_state.doc_status is None:
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px;'>
        <div style='font-size:64px'>🧠</div>
        <h2 style='color:#ececec; margin-top:16px'>DEV-DOCS RAG</h2>
        <p style='color:#666; font-size:16px'>Upload a PDF from the sidebar to start chatting</p>
        <p style='color:#444; font-size:13px'>Powered by Pinecone · Groq · HuggingFace</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.doc_status == "failed":
    st.error("Document processing failed. Upload a new document from the sidebar.")

elif st.session_state.doc_status == "ready":
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                render_citations(msg["sources"])

    if prompt := st.chat_input("Ask anything about your document..."):
        st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

        answer_placeholder = st.empty()
        full_answer   = ""
        final_sources = []

        with st.spinner(""):
            for token, sources in stream_answer(st.session_state.session_id, prompt):
                full_answer += token
                final_sources = sources if sources else final_sources
                answer_placeholder.markdown(
                    f'<div class="assistant-bubble">{full_answer}▌</div>',
                    unsafe_allow_html=True
                )

        answer_placeholder.markdown(
            f'<div class="assistant-bubble">{full_answer}</div>',
            unsafe_allow_html=True
        )
        render_citations(final_sources)

        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_answer,
            "sources": final_sources
        })
