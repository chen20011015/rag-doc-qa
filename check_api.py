"""API 连通性自检：配置完 Key 先跑这个，再启动 app。

用法：
    python check_api.py
    # 或显式传 key
    python check_api.py xxxxxxxx.xxxxxxxx
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def load_local_secrets() -> str | None:
    """读取 .streamlit/secrets.toml，模拟 Streamlit 的注入行为。

    这样本地自检和线上运行读的是同一份配置，不用两处维护。
    """
    path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if not path.exists():
        return None
    try:
        import tomllib
    except ImportError:
        return "需要 Python 3.11+ 才能自动读取 secrets.toml"

    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        return f"secrets.toml 解析失败：{exc}（检查引号和格式）"

    for key, value in data.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                os.environ.setdefault(str(sub_key), str(sub_value))
    return None


_secret_error = load_local_secrets()

from rag_core import (  # noqa: E402
    RagError,
    chat_config,
    chat_stream,
    embed_config,
    embed_texts,
)


def mask(key: str) -> str:
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}…{key[-4:]}（长度 {len(key)}）"


def main() -> int:
    if len(sys.argv) > 1:
        os.environ["API_KEY"] = sys.argv[1]

    if _secret_error:
        print(f"[配置错误] {_secret_error}")
        return 1

    chat_cfg = chat_config()
    embed_cfg = embed_config()

    print("=" * 52)
    print("配置检查")
    print("=" * 52)
    print(f"API_BASE    : {chat_cfg['base']}")
    print(f"CHAT_MODEL  : {chat_cfg['model']}")
    print(f"EMBED_MODEL : {embed_cfg['model']}")

    if not chat_cfg["key"]:
        print("\n[失败] 没有读到 API_KEY。")
        print("  本地：在 .streamlit/secrets.toml 写 API_KEY = \"xxx\"")
        print("  或：  python check_api.py 你的key")
        return 1

    print(f"API_KEY     : {mask(chat_cfg['key'])}")

    if not chat_cfg["key"].count(".") == 1:
        print("\n[提醒] 智谱的 Key 格式是 xxxxxxxx.xxxxxxxx（中间一个点）。")
        print("        如果这是 sk- 开头的 Key，多半是别的平台的，请换成智谱的。")

    failed = False

    print("\n[1/2] 测试向量模型 …")
    try:
        vectors = embed_texts(["连通性测试"])
        dim = vectors.shape[1]
        norm = float((vectors[0] ** 2).sum() ** 0.5)
        print(f"  [通过] 返回 {vectors.shape[0]} 条向量，维度 {dim}，模长 {norm:.3f}")
        if abs(norm - 1.0) > 0.05 and norm < 0.5:
            print("  [提醒] 向量模长异常，模型可能返回了非预期结果。")
    except RagError as exc:
        print(f"  [失败] {exc}")
        failed = True

    print("\n[2/2] 测试对话模型 …")
    if failed:
        print("  [跳过] 向量测试未通过，先解决上面的问题。")
    else:
        try:
            messages = [
                {"role": "system", "content": "你是测试助手，只回一句话。"},
                {"role": "user", "content": "用一句话说明你已就绪。"},
            ]
            text = "".join(chat_stream(messages))
            print(f"  [通过] 模型回复：{text.strip()[:60]}")
        except RagError as exc:
            print(f"  [失败] {exc}")
            failed = True

    print("\n" + "=" * 52)
    if failed:
        print("结果：存在问题，按上面的提示处理后重试。")
        print("常见原因：Key 复制不全 / Key 被禁用 / 账户余额为 0")
        return 1

    print("结果：配置正常，可以运行 streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
