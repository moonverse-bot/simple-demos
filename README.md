# AI 智能问答系统（Streamlit + DeepSeek + RAG）

一个基于 **Streamlit** 与 **DeepSeek API** 的 AI 应用项目，支持多模型切换、流式输出、数学公式渲染、多轮上下文记忆，并内置 **RAG 知识库问答**、**答案溯源**、**可切换检索方式** 与 **访问密码保护**。

## 功能亮点

- 🔐 访问密码保护：需输入密码才能使用，防止他人消耗 API 额度
- 💬 多轮对话：AI 能记住最近若干轮上下文
- ⚡ 真·流式输出：边生成边显示
- 🧠 多模型切换：DeepSeek Chat / DeepSeek Reasoner
- 📐 数学公式渲染：LaTeX 自动显示为漂亮分数
- 📚 RAG 知识库：上传文档，AI 依据文档内容回答
- 🔎 检索方式可切换：BM25 关键词 / ChromaDB 向量语义检索
- 🧷 答案溯源：引用资料时标注「（依据资料1）」来源，降低幻觉
- 🎛 参数可调：Temperature、Max Tokens、记忆轮数
- 🔧 工程化：API Key 走环境变量/Secrets、模块化设计

## 系统架构

```mermaid
flowchart LR
    U[用户] -->|输入密码| G[访问密码门]
    G -->|验证通过| F[Streamlit 前端 app.py]
    F -->|上传文档/检索| R[RAG 知识库 rag.py\nBM25 / ChromaDB 向量]
    R -->|相关片段 context| P[提示词构建 api_config.py]
    P -->|messages| M[DeepSeek API]
    M -->|流式回复| F
    F -->|标注依据资料| U
```

## 环境要求

- Python 3.9+
- DeepSeek API Key（环境变量 `DEEPSEEK_API_KEY`）

## 安装依赖

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

默认即可运行（BM25 检索，零额外依赖）。

**可选：启用 ChromaDB 向量检索（语义更准）**
```bash
pip install chromadb
```
安装后，在页面左侧「检索方式」选择「向量（ChromaDB）」即可；未安装时自动回退到 BM25。

## 配置 API Key 与访问密码

Windows PowerShell（临时，仅当前窗口生效）：
```powershell
$env:DEEPSEEK_API_KEY="sk-你的密钥"
$env:APP_PASSWORD="你的访问密码"
```

也可以写入系统环境变量，长期生效。

## 本地运行

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。若设置了 `APP_PASSWORD`，需先输入密码才能使用。

## 使用 RAG 知识库

1. 左侧「📚 知识库」区域上传 **txt / md / pdf** 文档
2. 提问时，系统自动检索相关片段并注入提示词
3. AI 会优先依据文档内容回答，引用时标注「（依据资料1）」来源

## 部署到 Streamlit Cloud（免费）

1. 把代码提交并推送到 GitHub：
   ```bash
   git add -A
   git commit -m "feat: 访问密码保护"
   git push origin main
   ```
2. 登录 [share.streamlit.io](https://share.streamlit.io)，点击 **New app**
3. 选择仓库 `moonverse-bot/simple-demos`、分支 `main`，入口文件填 `app.py`
4. 在 **Advanced settings → Secrets** 里添加（TOML）：
   ```
   DEEPSEEK_API_KEY = sk-你的密钥
   APP_PASSWORD = 你的访问密码
   ```
5. 点击 **Deploy**，等待构建完成即可获得公网访问链接

> 提醒：`APP_PASSWORD` 是保护你 DeepSeek 额度的关键。设置后，访客须先输入密码才能与 AI 对话。请把密码只发给需要体验的人。

## 项目结构

```
simple-demos/
├── app.py            # Streamlit 界面与交互（入口，含密码门）
├── api_config.py     # 模型调用、消息构建、公式渲染
├── rag.py            # 文档切分 + 检索（BM25 / ChromaDB 向量）
├── requirements.txt  # 依赖清单
└── README.md
```

## 说明

- 未配置 `APP_PASSWORD` 时不启用访问密码（开发模式）；生产/公网部署务必在 Secrets 中设置。
- 默认检索为 BM25（稀疏向量），轻量零依赖；安装 `chromadb` 后可切换为 Embedding 语义检索。
- 切换到推理模型 `deepseek-reasoner` 时，`Temperature` 参数无效（该模型不支持），属正常现象。
- API Key 请勿写进代码，统一从环境变量读取；Cloud 部署时在 Secrets 中配置。
