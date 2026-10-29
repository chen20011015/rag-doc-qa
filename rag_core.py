"""RAG 核心逻辑：文档解析 → 切分 → 向量化 → 检索 → 生成。

设计原则：
1. 只依赖 requests + numpy，没有重型框架，Streamlit 免费版也能跑起来。
2. 所有模型调用走 OpenAI 兼容协议，换厂商只改环境变量，不改代码。
3. 报错统一抛 RagError，带中文可读信息，前端直接展示给用户。
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import requests

DEFAULT_BASE = "https://open.bigmodel.cn/api/paas/v4"


class RagError(Exception):
    """可展示给用户的业务异常。"""


# ---------------------------------------------------------------- 配置


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def chat_config() -> dict:
    return {
        "base": _env("CHAT_API_BASE", "API_BASE", default=DEFAULT_BASE).rstrip("/"),
        "key": _env("CHAT_API_KEY", "API_KEY", "ZHIPU_API_KEY"),
        "model": _env("CHAT_MODEL", default="glm-4-flash"),
    }


def embed_config() -> dict:
    return {
        "base": _env("EMBED_API_BASE", "API_BASE", default=DEFAULT_BASE).rstrip("/"),
        "key": _env("EMBED_API_KEY", "API_KEY", "ZHIPU_API_KEY"),
        "model": _env("EMBED_MODEL", default="embedding-3"),
    }


# ---------------------------------------------------------------- 文档解析


@dataclass
class Chunk:
    """一个可检索片段。page 为 0 表示该格式不区分页码（txt/md/docx）。"""

    text: str
    source: str
    page: int
    order: int


def extract_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    """把上传的文件解析成 (页码, 文本) 列表。"""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RagError("缺少 pypdf 依赖，请执行 pip install -r requirements.txt") from exc
        try:
            reader = PdfReader(io.BytesIO(data))
        except Exception as exc:
            raise RagError(f"PDF 解析失败：{filename} 可能已损坏或加密。") from exc
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                pages.append((i, text))
        if not pages:
            raise RagError(
                f"{filename} 没有提取到文字，可能是扫描件（图片型 PDF），需要先做 OCR。"
            )
        return pages

    if ext in ("txt", "md", "markdown"):
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                return [(0, data.decode(enc))]
            except UnicodeDecodeError:
                continue
        raise RagError(f"{filename} 编码无法识别，请另存为 UTF-8 后重试。")

    if ext == "docx":
        try:
            import docx
        except ImportError as exc:  # pragma: no cover
            raise RagError("缺少 python-docx 依赖，请执行 pip install -r requirements.txt") from exc
        try:
            document = docx.Document(io.BytesIO(data))
        except Exception as exc:
            raise RagError(f"Word 解析失败：{filename}") from exc
        text = "\n".join(p.text for p in document.paragraphs)
        for table in document.tables:
            for row in table.rows:
                text += "\n" + " | ".join(c.text.strip() for c in row.cells)
        if not text.strip():
            raise RagError(f"{filename} 中没有文字内容。")
        return [(0, text)]

    raise RagError(f"暂不支持 .{ext} 格式，请上传 PDF / DOCX / TXT / MD。")


# ---------------------------------------------------------------- 切分

_SENTENCE = re.compile(r"[^。！？；!?;\n]+[。！？；!?;\n]?", re.S)
_WS = re.compile(r"[ \t\u3000]+")
_NL = re.compile(r"\n{3,}")


def _hard_split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def split_text(text: str, size: int = 480, overlap: int = 80) -> list[str]:
    """按中文句读切分，尽量保持句子完整；相邻片段保留 overlap 字重叠。"""
    if not text or not text.strip():
        return []
    cleaned = _NL.sub("\n\n", _WS.sub(" ", text)).strip()
    sentences = [s for s in _SENTENCE.findall(cleaned) if s.strip()]

    chunks: list[str] = []
    buffer: list[str] = []
    length = 0

    for sentence in sentences:
        if len(sentence) > size:
            if buffer:
                chunks.append("".join(buffer))
                buffer, length = [], 0
            chunks.extend(_hard_split(sentence, size))
            continue

        if length + len(sentence) > size:
            chunks.append("".join(buffer))
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(buffer):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            buffer, length = tail, tail_len

        buffer.append(sentence)
        length += len(sentence)

    if buffer:
        chunks.append("".join(buffer))

    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------- 接口调用


def _post(url: str, key: str, payload: dict, timeout: int = 60, retries: int = 3) -> dict:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_error: RagError | None = None

    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = RagError(f"网络请求失败：{exc}")
            time.sleep(1.5 * (attempt + 1))
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError as exc:
                raise RagError("接口返回非 JSON 内容，请检查 API 地址是否正确。") from exc

        if resp.status_code in (401, 403):
            raise RagError("API Key 无效或无权限（401/403），请检查 Secrets 配置。")

        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = RagError(f"接口限流或服务异常（{resp.status_code}）。")
            time.sleep(1.5 * (attempt + 1))
            continue

        raise RagError(f"接口报错 {resp.status_code}：{resp.text[:200]}")

    raise last_error or RagError("接口调用失败，请稍后重试。")


def embed_texts(
    texts: list[str],
    batch_size: int = 16,
    on_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """批量向量化，返回 shape=(n, dim) 的 float32 矩阵。"""
    cfg = embed_config()
    if not cfg["key"]:
        raise RagError("未配置 API Key：请在 Streamlit Secrets 中设置 API_KEY。")

    total = len(texts)
    vectors: list[np.ndarray] = []

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        payload = {"model": cfg["model"], "input": batch}
        data = _post(f"{cfg['base']}/embeddings", cfg["key"], payload)
        items = data.get("data") or []
        if len(items) != len(batch):
            raise RagError("向量化返回数量与输入不一致，请稍后重试。")
        items = sorted(items, key=lambda d: d.get("index", 0))
        for item in items:
            vectors.append(np.asarray(item["embedding"], dtype=np.float32))
        if on_progress:
            on_progress(min(start + batch_size, total), total)

    if not vectors:
        raise RagError("没有可向量化的文本。")

    matrix = np.vstack(vectors)
    if matrix.shape[1] == 0:
        raise RagError("向量维度异常，请检查 EMBED_MODEL 配置。")
    return matrix


class VectorIndex:
    """内存向量索引：归一化后做点积，等价于余弦相似度。"""

    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]):
        if vectors.shape[0] != len(chunks):
            raise RagError("向量数量与片段数量不一致。")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        self.matrix = (vectors / norms).astype(np.float32)
        self.chunks = chunks

    def search(self, query_vector: np.ndarray, k: int = 4, min_score: float = 0.0):
        norm = np.linalg.norm(query_vector) + 1e-9
        scores = self.matrix @ (query_vector / norm)
        order = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in order if scores[i] >= min_score]


# ---------------------------------------------------------------- 生成

SYSTEM_PROMPT = """你是一个严谨的文档问答助手，服务于企业知识库场景。

规则：
1. 只能依据【参考资料】作答，不得使用资料之外的知识，不得编造。
2. 每处关键结论后面标注来源编号，例如 [1]、[2]；多个来源写成 [1][3]。
3. 资料中没有答案时，明确回答：根据已上传的文档，我无法回答这个问题。并说明缺什么信息。
4. 用户问题与资料完全无关时，直接说明文档中未涉及，不要强行关联。
5. 使用简体中文，结论先行，能用要点就用要点，不要重复用户的问题。"""


def build_messages(
    question: str,
    hits: list[tuple[float, Chunk]],
    history: list[dict] | None = None,
) -> list[dict]:
    """拼装对话消息：系统提示 + 最近几轮上下文 + 本轮参考资料 + 问题。"""
    blocks = []
    for i, (score, chunk) in enumerate(hits, start=1):
        where = f"{chunk.source}" + (f" 第 {chunk.page} 页" if chunk.page else "")
        blocks.append(f"【资料{i}】来源：{where}（相关度 {score:.2f}）\n{chunk.text}")

    user_content = (
        "【参考资料】\n"
        + "\n\n".join(blocks)
        + f"\n\n【问题】\n{question}"
    )

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in (history or [])[-6:]:
        messages.append({"role": item["role"], "content": item["content"][:800]})
    messages.append({"role": "user", "content": user_content})
    return messages


def chat_stream(messages: list[dict], timeout: int = 180) -> Iterator[str]:
    """流式调用对话接口，逐段 yield 文本。"""
    cfg = chat_config()
    if not cfg["key"]:
        raise RagError("未配置 API Key：请在 Streamlit Secrets 中设置 API_KEY。")

    url = f"{cfg['base']}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg['key']}", "Content-Type": "application/json"}
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.2,
        "stream": True,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
    except requests.RequestException as exc:
        raise RagError(f"网络请求失败：{exc}") from exc

    if resp.status_code != 200:
        detail = resp.text[:200]
        if resp.status_code in (401, 403):
            raise RagError("API Key 无效或无权限（401/403），请检查 Secrets 配置。")
        raise RagError(f"对话接口报错 {resp.status_code}：{detail}")

    emitted = False
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        content = (choices[0].get("delta") or {}).get("content")
        if content:
            emitted = True
            yield content

    if not emitted:
        raise RagError("模型没有返回内容，可能是额度不足或触发内容审核，请稍后重试。")
