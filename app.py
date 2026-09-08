import streamlit as st
from api_config import (
    build_messages,
    stream_ai_reply,
    render_latex,
    AVAILABLE_MODELS,
    DEFAULT_SYSTEM_PROMPT,
)
from rag import make_rag, chroma_available

# 页面配置必须放在最顶部
st.set_page_config(
    page_title="AI对话助手",
    page_icon="🤖",
    layout="wide"
)

# ---------- 初始化状态 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "bm25"          # bm25 / vector
if "rag" not in st.session_state:
    st.session_state.rag = make_rag("bm25")
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = set()


def get_file_text(filename, data) -> str:
    """读取上传文件内容：txt/md 直接解码，pdf 用 pypdf。"""
    if filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except ImportError:
            st.sidebar.warning("解析 PDF 需要安装 pypdf：pip install pypdf")
            return ""
    return data.decode("utf-8", errors="ignore")


# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ 设置面板")

    model_label = st.selectbox("选择模型", list(AVAILABLE_MODELS.keys()))
    model = AVAILABLE_MODELS[model_label]
    st.caption(f"当前接口模型：`{model}`")

    temp = st.slider("创造力 (Temperature)", 0.0, 1.0, 0.7, 0.05)
    max_len = st.slider("回答最大长度 (Max Tokens)", 200, 1500, 600, 100)
    max_rounds = st.slider("记忆对话轮数", 1, 20, 10, 1)

    # ---------- 检索方式 ----------
    mode_label = st.selectbox("检索方式", ["BM25 关键词", "向量（ChromaDB）"])
    mode = "vector" if mode_label.startswith("向量") else "bm25"
    if mode == "vector" and not chroma_available:
        st.sidebar.warning("未安装 chromadb，已回退到 BM25。可运行 pip install chromadb 启用向量检索。")
        mode = "bm25"
    if st.session_state.rag_mode != mode:
        st.session_state.rag = make_rag(mode)
        st.session_state.rag_mode = mode
        if st.session_state.uploaded_names:
            st.session_state.uploaded_names = set()
            st.sidebar.info("切换检索方式后，请重新上传知识库文档。")

    st.divider()

    # ---------- 知识库（RAG） ----------
    st.subheader("📚 知识库")
    uploaded = st.file_uploader(
        "上传资料（txt / md / pdf）",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        for f in uploaded:
            if f.name in st.session_state.uploaded_names:
                continue
            content = get_file_text(f.name, f.getvalue())
            if content.strip():
                try:
                    st.session_state.rag.add_document(f.name, content)
                    st.session_state.uploaded_names.add(f.name)
                    st.sidebar.success(f"已加载：{f.name}")
                except Exception as e:
                    st.sidebar.error(f"加载失败：{e}")

    rag = st.session_state.rag
    st.caption(f"已加载：{len(st.session_state.uploaded_names)} 个文档，共 {rag.doc_count} 个片段")

    if st.button("🗑️ 清空知识库", disabled=rag.is_empty):
        rag.clear()
        st.session_state.uploaded_names = set()
        st.rerun()

    st.divider()

    if st.button("🗑️ 清空对话", type="primary"):
        st.session_state.messages = []
        st.rerun()

# ---------- 主界面 ----------
st.title("🤖 AI在线交流对话")
st.divider()

# 渲染历史对话（对公式做一次 LaTeX 渲染）
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(render_latex(msg["content"]))

# 输入框
user_text = st.chat_input("输入问题和AI对话...")

# ---------- 处理用户输入 ----------
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # 先检索知识库，把相关片段注入到系统提示词
    custom_system = DEFAULT_SYSTEM_PROMPT
    if not st.session_state.rag.is_empty:
        context = st.session_state.rag.build_context(user_text, k=4)
        if context:
            custom_system += (
                "\n\n以下是用户提供的文档资料，请优先依据这些资料回答；"
                "如果引用了资料内容，请在回答末尾标注来源编号，例如（依据资料1）。"
                "如果资料中没有答案，再凭通用知识回答，并说明参考了资料。\n\n"
                + context
            )

    with st.chat_message("assistant"):
        messages = build_messages(
            history=st.session_state.messages,
            system_prompt=custom_system,
            max_history_pairs=max_rounds
        )
        full_ans = st.write_stream(
            stream_ai_reply(
                messages=messages,
                model=model,
                temperature=temp,
                max_tokens=max_len
            )
        )
        full_ans = render_latex(full_ans)

    st.session_state.messages.append({"role": "assistant", "content": full_ans})
