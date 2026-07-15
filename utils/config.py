import os
from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ChromaDB + BGE-M3（免费，中文效果更好）
collection_name = "knowledge-chroma"
persist_directory = "db/chroma_db"
bge_model = "BAAI/bge-m3"

# MinerU 配置
MINERU_ENABLED = os.getenv("MINERU_ENABLED", "true").lower() == "true"
text_path = os.path.join(BASE_DIR, "knowledge_data")

# 文本切割
chunk_size = 500
chunk_overlap = 100
separators = ["\n\n", "\n", "。", "；", "，", " ", ""]
max_splter_char_number = 1000

# 检索开关
USE_QUERY_REWRITE = True
USE_HYBRID_SEARCH = True
USE_RERANK = True

# MD5去重
md5_path = os.path.join(BASE_DIR, "data", "md5_records.txt")

# Cross-Encoder
cross_encoder_model = "BAAI/bge-reranker-v2-m3"

# 检索参数
top_k = 3
vector_top_k = 20
bm25_top_k = 20
rerank_top_k = 15
vector_weight = 0.5
bm25_weight = 0.5
final_top_k = 3