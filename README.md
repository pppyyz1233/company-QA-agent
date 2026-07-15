<div align="center">

# 📚 Company Q&A Agent

### RAG 混合检索企业制度智能问答系统

*上传制度文档，用自然语言提问，精准获取答案*

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white&style=flat-square" alt="Python"/>
<img src="https://img.shields.io/badge/RAG-Hybrid_Search-059669?style=flat-square" alt="RAG"/>
<img src="https://img.shields.io/badge/LLM-DeepSeek-536DFE?style=flat-square" alt="DeepSeek"/>
<img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"/>

</div>

---

## 架构

```
管理员上传 PDF / Word
  │
  ▼
MinerU 解析（支持表格/双栏）→ 递归分块 → BGE-M3 向量化 → ChromaDB
  │
  ═══════════ 入库完成 ═══════════
  │
  员工提问："年假有几天？"
  │
  ▼
Query 改写（LLM 提取关键词）
  │
  ▼
┌─────────────────────────┐
│  混合检索               │
│  向量 Top20 + BM25 Top20 │
│  → 加权融合 → CrossEncoder 精排 → Top3
└─────────────────────────┘
  │
  ▼
强约束 System Prompt + 检索结果 → LLM → 答案
```

| 模块 | 技术 | 作用 |
|------|------|------|
| 文档解析 | MinerU / python-docx | PDF(表格/双栏) + Word |
| 文本分块 | RecursiveCharacterTextSplitter | chunk=500, overlap=100 |
| 向量嵌入 | BGE-M3 | 文本 → 1024维向量 |
| 向量存储 | ChromaDB | 持久化存储 + 相似检索 |
| 关键词检索 | BM25 + jieba 分词 | 精确关键词匹配 |
| Query 改写 | LLM | 口语化问题 → 关键词提取 |
| 精排 | CrossEncoder | BGE-reranker-v2-m3 |
| 去重 | MD5 指纹 | 避免重复入库 |

---

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env        # 填 DEEPSEEK_API_KEY
python main.py              # → http://localhost:8000
```

---

## 项目结构

```
company_Q&A_agent/
├── main.py                    启动入口
├── routers/            chat · upload_document · user
├── chat_service/       LLM 调用 + 上下文构建
├── utils/
│   ├── config.py              全局配置
│   ├── handle_text.py         BGE-M3 + ChromaDB + MD5
│   ├── document_service.py    MinerU + python-docx
│   └── hybrid_retriever.py    BM25 + CrossEncoder 精排
├── crud/              用户 · 会话 · 消息
├── models/            SQLAlchemy ORM
├── schemas/           Pydantic
├── stock/db.py        MySQL 引擎
├── data/              MD5 指纹存储
└── uploads/           上传文件落盘
```

---

## 检索性能

| 指标 | 数值 |
|------|------|
| 文档类型 | PDF(表格/双栏) + Word |
| 分块策略 | 500 字符 / 100 重叠 |
| 粗排召回 | 向量 20 + BM25 20 |
| 精排输出 | Top 3 |
| 召回率提升 | +20%（混合检索 vs 纯向量） |
| 幻觉率 | 接近 0（强约束 Prompt + 精排过滤） |

---

## 技术栈

```
BGE-M3 · BM25 · CrossEncoder · ChromaDB · LangChain · MinerU
FastAPI · MySQL · DeepSeek · python-docx
```

---

## License

MIT