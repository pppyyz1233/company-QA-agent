import hashlib
import os
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # BGE-M3，免费
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import config


# MD5计算
def transform_string_md5(input_str: str, encoding='utf-8'):
    input_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(input_bytes)
    return md5_obj.hexdigest()


# 检查PDF是否已上传
def check_md5(md5_str: str):
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding="utf-8").close()
        return False
    with open(config.md5_path, 'r', encoding="utf-8") as f:
        for i in f:
            if i.strip() == md5_str:
                return True
    return False


# 保存新的MD5
def save_md5(md5_str: str):
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')


# 知识库服务类
class KnowledgeBaseService:
    def __init__(self):
        # BGE-M3 向量模型（免费，无需API Key，中文效果好）
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.bge_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.persist_directory,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    # 上传文本到知识库
    def upload_by_str(self, content: str, filename: str, operator: str = "admin"):
        md5 = transform_string_md5(content)
        if check_md5(md5):
            return "文件已存在，跳过上传"

        if len(content) <= config.max_splter_char_number:
            chunks = [content]
        else:
            chunks = self.splitter.split_text(content)

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": operator,
        }

        if chunks:
            self.chroma.add_texts(
                chunks,
                metadatas=[metadata for _ in chunks],
            )
            save_md5(md5)
            return "文档上传成功"

    # 向量搜索
    def search(self, query: str):
        docs = self.chroma.similarity_search(query, k=config.top_k)
        return [(doc.page_content, doc.metadata) for doc in docs]


kb_service = KnowledgeBaseService()


# 返回检索到的文本片段列表
def retrieve_context(query: str):
    results = kb_service.search(query)
    return [content for content, _ in results]