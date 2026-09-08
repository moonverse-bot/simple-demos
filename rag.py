"""
简单 RAG（检索增强）模块：文档切分 + BM25 检索。

- 支持 txt / md / pdf（pdf 需安装 pypdf）。
- 默认使用 jieba 分词（可选）；未安装时退化为中英文字符/单词切分。
- 检索得到的片段会注入提示词，让 AI 基于文档回答。
"""

import re
from collections import Counter

try:
    import jieba
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False


def tokenize(text: str):
    """分词：优先 jieba，否则退化为中英文粗切。"""
    text = text.lower()
    if _HAS_JIEBA:
        return [w for w in jieba.lcut(text) if w.strip()]
    words = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text)
    return [w for w in words if w.strip()]


def _chunk(text: str, size: int = 300, overlap: int = 60):
    """按字符切分，带重叠，避免切断语义。"""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += size - overlap
    return chunks


class SimpleRAG:
    """轻量内存版知识库，支持添加文档、检索与拼接上下文。"""

    def __init__(self):
        self.docs = []          # 每个片段：{"source": 来源, "text": 内容, "chunk_id": 序号}
        self._tokenized = []    # 每个片段的词表
        self.avg_len = 0.0

    @property
    def is_empty(self) -> bool:
        return len(self.docs) == 0

    @property
    def doc_count(self) -> int:
        return len(self.docs)

    def add_document(self, name: str, text: str):
        """把一个文档切分后加入知识库。"""
        for chunk in _chunk(text):
            self.docs.append({
                "source": name,
                "text": chunk,
                "chunk_id": len(self.docs) + 1,
            })
            self._tokenized.append(tokenize(chunk))
        n = len(self._tokenized)
        self.avg_len = (sum(len(x) for x in self._tokenized) / n) if n else 0.0

    def clear(self):
        self.docs = []
        self._tokenized = []
        self.avg_len = 0.0

    def retrieve(self, query: str, k: int = 4):
        """返回与 query 最相关的 k 个片段。"""
        if not self.docs:
            return []
        q = tokenize(query)
        scores = [self._bm25(q, i) for i in range(len(self._tokenized))]
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in top if scores[i] > 0]

    def _bm25(self, query_tokens, doc_idx: int) -> float:
        doc = self._tokenized[doc_idx]
        N = len(self._tokenized)
        k1, b = 1.5, 0.75
        df = Counter()
        for token in query_tokens:
            df[token] = sum(1 for d in self._tokenized if token in d)
        score = 0.0
        dl = len(doc)
        for token in query_tokens:
            tf = doc.count(token)
            if tf == 0:
                continue
            idf = max(0.0, (N - df[token] + 0.5) / (df[token] + 0.5))
            denom = tf + k1 * (1 - b + b * dl / (self.avg_len or 1))
            score += idf * (tf * (k1 + 1)) / denom
        return score

    def build_context(self, query: str, k: int = 4) -> str:
        """把检索到的片段拼成可直接注入提示词的文本。"""
        hits = self.retrieve(query, k)
        if not hits:
            return ""
        lines = []
        for i, h in enumerate(hits, 1):
            lines.append(f"[资料{i}·来源：{h['source']}] {h['text']}")
        return "\n\n".join(lines)
