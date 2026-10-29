"""文档智能问答 · Streamlit 前端。

流程：上传文档 → 解析切分 → 向量化建索引 → 检索 → 带引用生成。
"""

from __future__ import annotations

import os

import streamlit as st

# 把 Streamlit Secrets 注入环境变量，rag_core 统一从环境变量读取配置。
# 本地调试时也可以用 export API_KEY=xxx 直接注入。
def _load_secrets() -> None:
    try:
        secrets = st.secrets
    except Exception:
        return
    for key, value in secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                os.environ.setdefault(str(sub_key), str(sub_value))


_load_secrets()

from rag_core import (  # noqa: E402
    Chunk,
    RagError,
    VectorIndex,
    build_messages,
    chat_config,
    chat_stream,
    embed_config,
    embed_texts,
    extract_pages,
    split_text,
)

MAX_FILE_MB = 20
SUPPORTED = ["pdf", "docx", "txt", "md"]

st.set_page_config(page_title="文档智能问答 · RAG", page_icon="📚", layout="wide")


# ---------------------------------------------------------------- 索引构建


def build_index(files, chunk_size: int, overlap: int) -> VectorIndex:
    bar = st.sidebar.progress(0.0, text="解析文档…")
    chunks: list[Chunk] = []

    for i, file in enumerate(files):
        data = file.getvalue()
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            bar.empty()
            raise RagError(f"{file.name} 超过 {MAX_FILE_MB}MB，请拆分后再上传。")
        for page, text in extract_pages(file.name, data):
            for piece in split_text(text, chunk_size, overlap):
                chunks.append(
                    Chunk(text=piece, source=file.name, page=page, order=len(chunks))
                )
        bar.progress(0.4 * (i + 1) / len(files), text=f"解析 {file.name}…")

    if not chunks:
        bar.empty()
        raise RagError("没有从文档中提取到可用文本。")

    def on_progress(done: int, total: int) -> None:
        bar.progress(
            0.4 + 0.55 * done / total, text=f"向量化 {done}/{total} 个片段…"
        )

    vectors = embed_texts([c.text for c in chunks], on_progress=on_progress)
    bar.progress(1.0, text="索引构建完成")
    bar.empty()
    return VectorIndex(vectors, chunks)


def file_signature(files) -> tuple:
    return tuple(sorted((f.name, f.size) for f in files))


# ---------------------------------------------------------------- 展示


def render_sources(hits) -> None:
    if not hits:
        return
    with st.expander(f"📎 引用的原文片段（{len(hits)} 条）"):
        for i, (score, chunk) in enumerate(hits, start=1):
            where = chunk.source + (f" · 第 {chunk.page} 页" if chunk.page else "")
            st.markdown(f"**[{i}]** `{where}` · 相关度 {score:.2f}")
            st.caption(chunk.text[:400] + ("…" if len(chunk.text) > 400 else ""))
            if i < len(hits):
                st.divider()


# ---------------------------------------------------------------- 侧边栏

with st.sidebar:
    st.markdown("### 📚 文档智能问答")
    st.caption("上传文档，基于内容提问，答案可追溯到原文")

    st.markdown("**① 上传文档**")
    uploaded = st.file_uploader(
        "支持 PDF / DOCX / TXT / MD，可多选",
        type=SUPPORTED,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    with st.expander("⚙️ 检索参数", expanded=False):
        top_k = st.slider("召回片段数", 2, 8, 4)
        chunk_size = st.slider("片段长度（字）", 200, 800, 480, step=40)
        overlap = st.slider("片段重叠（字）", 0, 200, 80, step=20)
        min_score = st.slider("最低相关度", 0.0, 0.8, 0.30, step=0.05)

    build_disabled = not uploaded
    col_a, col_b = st.columns(2)
    if col_a.button("重建索引", use_container_width=True, disabled=build_disabled):
        st.session_state.pop("signature", None)
        st.rerun()
    if col_b.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    chat_cfg = chat_config()
    embed_cfg = embed_config()
    if chat_cfg["key"] and embed_cfg["key"]:
        st.success("API 已配置", icon="✅")
    else:
        st.error("未检测到 API Key，请在 Secrets 中配置 API_KEY", icon="⚠️")
    st.caption(f"对话模型：`{chat_cfg['model']}` · 向量模型：`{embed_cfg['model']}`")

    if "index" in st.session_state:
        st.metric("已索引片段", len(st.session_state.index.chunks))

    st.divider()
    st.caption("本 Demo 由 Saiq 构建 · 数据仅在内存中处理，刷新即清空")

# ---------------------------------------------------------------- 自动建索引

st.session_state.setdefault("messages", [])

if uploaded:
    signature = file_signature(uploaded)
    if st.session_state.get("signature") != signature or "index" not in st.session_state:
        try:
            with st.spinner("正在建立索引，首次上传需要几十秒…"):
                st.session_state.index = build_index(uploaded, chunk_size, overlap)
            st.session_state.signature = signature
            st.session_state.messages = []
        except RagError as exc:
            st.session_state.pop("index", None)
            st.error(str(exc))
            st.stop()

# ---------------------------------------------------------------- 主区域

st.title("文档智能问答")
st.caption("RAG 全流程 Demo：文档解析 → 文本切分 → 向量检索 → 带引用生成")

index: VectorIndex | None = st.session_state.get("index")

if index is None:
    st.info("### 开始使用\n1. 在左侧上传 PDF / DOCX / TXT / MD 文档\n2. 等待自动建立索引\n3. 在下方提问，答案会标注原文出处")
    col1, col2, col3 = st.columns(3)
    col1.metric("支持格式", "PDF / DOCX / TXT")
    col2.metric("答案可溯源", "逐条引用")
    col3.metric("单文件上限", f"{MAX_FILE_MB} MB")
    st.markdown(
        "**技术栈**：`Streamlit` · `pypdf` · `NumPy 向量检索` · `OpenAI 兼容 API`"
    )
    st.stop()

sources = sorted({c.source for c in index.chunks})
st.markdown(
    f"当前知识库：**{len(sources)} 个文档 / {len(index.chunks)} 个片段** ｜ "
    + " · ".join(f"`{s}`" for s in sources[:5])
    + (" 等" if len(sources) > 5 else "")
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("hits"):
            render_sources(message["hits"])

question = st.chat_input("就文档内容提问，例如：这个流程的验收标准是什么？")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("检索中…"):
                query_vector = embed_texts([question])[0]
                hits = index.search(query_vector, k=top_k, min_score=min_score)

            if not hits:
                answer = "根据已上传的文档，我无法回答这个问题。建议确认文档是否包含相关内容，或换一种问法。"
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                history = [
                    m for m in st.session_state.messages[:-1] if m["role"] in ("user", "assistant")
                ]
                messages = build_messages(question, hits, history)
                answer = st.write_stream(chat_stream(messages))
                render_sources(hits)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "hits": hits}
                )
        except RagError as exc:
            st.error(str(exc))
