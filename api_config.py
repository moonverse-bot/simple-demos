import os
import re
from openai import OpenAI

api_key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
if not api_key:
    raise ValueError("❌ 请设置环境变量 DEEPSEEK_API_KEY")

# 默认接入 DeepSeek，可通过环境变量覆盖以接入其他 OpenAI 兼容服务
base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com').strip()

client = OpenAI(api_key=api_key, base_url=base_url)

# 支持的大模型列表：界面显示名 -> 接口模型名
AVAILABLE_MODELS = {
    "DeepSeek Chat（通用对话）": "deepseek-chat",
    "DeepSeek Reasoner（推理）": "deepseek-reasoner",
}

DEFAULT_SYSTEM_PROMPT = (
    "你是一个乐于助人的 AI 助手，请用简洁、清晰的中文回答用户的问题。"
    "当回答包含数学公式时，请使用 LaTeX 语法，并用 $...$ 包裹行内公式、"
    "用 $$...$$ 包裹独立成行的公式。例如：一又三分之二写作 $1\\frac{2}{3}$。"
)


def render_latex(text: str) -> str:
    """
    把文本中未加 $ 包裹的 LaTeX 分数（\\frac{...}{...}）自动用 $...$ 包裹，
    这样 Streamlit 的 Markdown 才能渲染成漂亮的数字形式。
    """
    pattern = re.compile(r'(?<![\$\\])\\frac\{[^{}]*\}\{[^{}]*\}')

    def repl(m):
        return '$' + m.group(0) + '$'

    return pattern.sub(repl, text)


def build_messages(
    history: list,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_history_pairs: int = 15
) -> list:
    """把会话历史整理成符合 OpenAI 格式的消息列表，只保留最近 N 轮。"""
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-max_history_pairs * 2:])
    return messages


def _common_params(messages, temperature, max_tokens, model, stream):
    """构造请求参数；推理模型不支持 temperature，按模型决定是否传。"""
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if model != "deepseek-reasoner":
        params["temperature"] = temperature
    return params


def get_ai_reply(
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 600,
    model: str = "deepseek-chat"
) -> str:
    """非流式：一次性返回完整回复。"""
    try:
        response = client.chat.completions.create(
            **_common_params(messages, temperature, max_tokens, model, stream=False)
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ API 调用出错：{e}"


def stream_ai_reply(
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 600,
    model: str = "deepseek-chat"
):
    """流式：边生成边产出文本块，供前台实时显示。"""
    try:
        response = client.chat.completions.create(
            **_common_params(messages, temperature, max_tokens, model, stream=True)
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"❌ API 调用出错：{e}"
