"""
简单 RAG 模块：支持两种检索方式。

1. SimpleRAG  —— 基于 BM25 的关键词检索（零依赖，默认保底）。
2. ChromaRAG —— 基于 ChromaDB + Embedding 的语义向量检索（可选，需安装 chromadb）。

两个类接口一致：add_document / build_context / clear / is_empty / doc_count。
"""

import re
from collections import Counter

try:
    import jieba
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False

try:
    import chromadb
    _HAS_CHROMA = True
except Exception:
    _HAS_CHROMA = False

# 供 UI 判断向量检索是否可用
chroma_available = _HAS_CHROMA


def tokenize(text: str):
    """分词：优先 jieba，否则退化为中英文字符/单词切分。"""
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
    """BM25 稀疏检索：轻量、零依赖，作为默认保底方案。"""

    def __init__(self):
        self.docs = []
        self._tokenized = []
        self.avg_len = 0.0

    @property
    def is_empty(self) -> bool:
        return len(self.docs) == 0

    @property
    def doc_count(self) -> int:
        return len(self.docs)

    def add_document(self, name: str, text: str):
        for chunk in _chunk(text):
            self.docs.append({"source": name, "text": chunk, "chunk_id": len(self.docs) + 1})
            self._tokenized.append(tokenize(chunk))
        n = len(self._tokenized)
        self.avg_len = (sum(len(x) for x in self._tokenized) / n) if n else 0.0

    def clear(self):
        self.docs = []
        self._tokenized = []
        self.avg_len = 0.0

    def retrieve(self, query: str, k: int = 4):
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
        hits = self.retrieve(query, k)
        if not hits:
            return ""
        lines = []
        for i, h in enumerate(hits, 1):
            lines.append(f"[资料{i}·来源：{h['source']}] {h['text']}")
        return "\n\n".join(lines)


class ChromaRAG:
    """ChromaDB 向量检索：基于 Embedding 的语义检索，需安装 chromadb。"""

    def __init__(self, collection_name: str = "kb"):
        if not _HAS_CHROMA:
            raise RuntimeError("未安装 chromadb，无法使用向量检索")
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._ids = 0

    @property
    def is_empty(self) -> bool:
        try:
            return self.collection.count() == 0
        except Exception:
            return True

    @property
    def doc_count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def add_document(self, name: str, text: str):
        chunks = _chunk(text)
        if not chunks:
            return
        ids, docs, metas = [], [], []
        for c in chunks:
            self._ids += 1
            ids.append(str(self._ids))
            docs.append(c)
            metas.append({"source": name})
        self.collection.add(ids=ids, documents=docs, metadatas=metas)

    def clear(self):
        try:
            self.client.delete_collection("kb")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="kb", metadata={"hnsw:space": "cosine"}
        )
        self._ids = 0

    def build_context(self, query: str, k: int = 4) -> str:
        if self.is_empty:
            return ""
        try:
            n = self.doc_count
            res = self.collection.query(
                query_texts=[query], n_results=min(k, n)
            )
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            lines = []
            for i, (doc, meta) in enumerate(zip(docs, metas), 1):
                src = meta.get("source", "资料") if meta else "资料"
                lines.append(f"[资料{i}·来源：{src}] {doc}")
            return "\n\n".join(lines)
        except Exception:
            return ""


def make_rag(mode: str):
    """工厂函数：根据检索方式返回对应实例；向量不可用时回退 BM25。"""
    if mode == "vector" and _HAS_CHROMA:
        try:
            return ChromaRAG()
        except Exception:
            pass
    return SimpleRAG()
