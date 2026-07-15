from typing import Optional, List, Tuple, Dict

import jieba
from openai import OpenAI

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from utils import config
from utils.config import bm25_top_k, rerank_top_k, vector_top_k, final_top_k


class QueryRewriter:

    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )

        self.model = "deepseek-chat"

    def _call_llm(self, prompt: str):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[Query改写] LLM 调用失败: {e}")
            return ""

    def rewrite(self, query:str):
        prompt =f"""你是一个查询改写助手。请将用户的自然语言问题改写成适合在文档库中进行语义检索的关键词或短句。
        规则：
1. 提取核心概念和关键词
2. 去除口语化表达（如"我想问一下"、"帮我查查"）
3. 保留专业术语和数字
4. 只输出改写后的查询，不要加任何解释

用户问题：{query}

改写后的查询：
        """
        rewritten = self._call_llm(prompt)
        if rewritten:
            print(f"[Query改写] {query[:30]}... → {rewritten[:30]}...")
            return rewritten
        else:
            print(f"[Query改写] 失败，使用原查询")
            return query


class BM25Retriever:
    def __init__(self):
        self.bm25:Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.initialized = False

    def _tokenize(self, text:str) :
        try:
            return list(jieba.cut(text))
        except ImportError:
            return list(text)

#加载chroma文件
    def load_from_chroma(self, chroma_client):
        try:
            result = chroma_client.get()
            if not result or not result.get("documents"):
                print("[BM25] Chroma 中没有文档，请先上传")
                return False

            self.documents = result["documents"]

            tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            self.initialized = True
            return True

        except Exception as e:
            print(f"[BM25] 从 Chroma 加载失败: {e}")
            return False

    def search(self, query:str):
        tokenized_query = self._tokenize(query)
        score = self.bm25.get_scores(tokenized_query)

        indeed_score=list(enumerate(score))
        indeed_score.sort(key=lambda x: x[1], reverse=True)

        max_score=indeed_score[0][1] if indeed_score else 0

        result = []
        for idx, score in indeed_score[:bm25_top_k]:
            if score > 0:
                normalized_score = score / max_score
                result.append((self.documents[idx], normalized_score))

        return result

class Cross_Encoder:
    def __init__(self):
        self.model_name = config.cross_encoder_model
        self.model: Optional[CrossEncoder] = None

    def _load(self):
        if self.model is None:
            self.model = CrossEncoder(self.model_name,max_length=512)

    def _rerank(
            self,
            query:str,
            candidates:List[str],
    ):
        self._load()
        if not candidates:return []

        pairs = [[query, doc] for doc in candidates]
        score = self.model.predict(pairs)

        ranked_candidates = zip(candidates, score)
        ranked = sorted(ranked_candidates,key=lambda x: x[1], reverse=True)

        return [(doc, float(score)) for doc, score in ranked[:rerank_top_k]]


class RetrievalOptimizer:
    def __init__(self, chroma_client):
        self.chroma = chroma_client

        self.query_rewriter = QueryRewriter()
        self.bm25_retriever = BM25Retriever()
        self.rerank = Cross_Encoder()

        self.bm25_retriever.load_from_chroma(chroma_client)

    def _chroma_search(self, query: str, top_k: int = None):
        if top_k is None:
            top_k = vector_top_k
        results = self.chroma.similarity_search_with_relevance_scores(query, top_k)
        return [(doc.page_content, score) for doc, score in results]

    def _merge_rankings(
            self,
            chroma_results: List[Tuple[str, float]],
            bm25_results: List[Tuple[str, float]],
    ):
        doc_scores: Dict[str, dict] = {}

        for content, score in chroma_results:
            key = content[:200]  #前200字符作为去重 key
            if key not in doc_scores:
                doc_scores[key] = {"content": content, "score": 0.0}
            doc_scores[key]["score"] += score * config.vector_weight

        for content, score in bm25_results:
            key = content[:200]
            if key not in doc_scores:
                doc_scores[key] = {"content": content, "score": 0.0}
            doc_scores[key]["score"] += score * config.bm25_weight

        sorted_items = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        return [item["content"] for item in sorted_items]

    def retrieve(
            self,
            query: str,
            use_query_rewrite: bool = None,
            use_hybrid: bool = None,
            use_rerank: bool = None,
    ):
        search_query = query
        if use_query_rewrite:
            search_query = self.query_rewriter.rewrite(query)

        if use_hybrid:
            chroma_results = self._chroma_search(search_query)
            bm25_results = self.bm25_retriever.search(search_query)

            merged = self._merge_rankings(chroma_results, bm25_results)

            if use_rerank and len(merged) > final_top_k:
                candidates = merged[:config.rerank_top_k]
                reranked = self.rerank._rerank(search_query, candidates, final_top_k)
                docs = [doc for doc, score in reranked]
            else:
                docs = merged[:final_top_k]
        else:
            chroma_results = self._chroma_search(search_query, final_top_k)
            docs = [content for content, score in chroma_results]

        print(f"{'=' * 60}\n")
        return docs


_optimizer: Optional[RetrievalOptimizer] = None


def get_optimizer(chroma_client=None) -> Optional[RetrievalOptimizer]:
    global _optimizer

    if _optimizer is None and chroma_client is not None:
        _optimizer = RetrievalOptimizer(chroma_client)

    return _optimizer


def optimized_retrieve(
        query: str,
        chroma_client=None,
        **kwargs,
):
    optimizer = get_optimizer(chroma_client)

    if optimizer is None:
        return []

    return optimizer.retrieve(query, **kwargs)





