import traceback

from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from crud import message, conversation
from utils import config
from utils.handle_text import retrieve_context, kb_service
from utils.hybrid_retriever import optimized_retrieve


#调用DeepSeek大模型
def call_deepseek(messages: list):
    deepseek_llm =OpenAI(
        api_key = "sk-097c5cc5fb5c461cb0742215809358ba",
        base_url = config.DEEPSEEK_BASE_URL,
    )

    try:
        response = deepseek_llm.chat.completions.create(
            model = "deepseek-chat",
            messages = messages,
            temperature = 0.3,
            max_tokens = 2000,
        )
        return response.choices[0].message.content

    except Exception :
        return "调用大模型失败"

#获取或创建会话
async def get_or_create_conversation(
        db: AsyncSession,
        user_id: int,
        conversation_id:int,
        title:str ="新对话"
):
    is_new = False

    if conversation_id :
        conversations = await conversation.get_conversation_by_id(db,conversation_id, user_id )
        if not conversations:
            raise ValueError("对话不存在")

    else:
        conversations = await conversation.create_conversation(db, user_id, title)
        is_new = True

    return conversations,is_new


#知识库检索
async def retrieve_relevant_context(question: str):
     try:
        docs = optimized_retrieve(
            query=question,
            chroma_client=kb_service.chroma,
        )
        context =  "\n\n".join(docs) if docs else "未找到相关制度"
        sources = docs if docs else []

        return context, sources

     except Exception:
         print("优化检索失败，降级为原始检索")
         try:
             docs = retrieve_context(question)
             context = "\n\n".join(docs) if docs else "未找到相关制度"
             sources = docs if docs else []
             return context, sources
         except Exception:
             return "未找到相关制度", []


#构建提示词
def build_system_prompt(context: str):
    return f"""你是企业制度助手。请严格根据以下制度内容回答用户问题。
如果制度中没有提到相关内容，请回答"制度中未提及"，不要编造答案。

制度内容：
{context}
"""


#
async def process_question(
        db: AsyncSession,
        user_id: int,
        question: str,
        conversation_id:int = None,
):
    try:
        conversation,is_new =await get_or_create_conversation(db, user_id,conversation_id)

        await message.save_message(db, conversation.id, "user", question)

        context,sources = await retrieve_relevant_context(question)

        systemp_prompt = build_system_prompt(context)
        messages =[
            {"role":"system","content":systemp_prompt},
            {"role":"system","content":question}
        ]

        answer = call_deepseek(messages)

        saved_message = await message.save_message( db, conversation.id, "assistant", answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "conversation_id": conversation.id,
            "message_id": saved_message.id,
            "is_new_conversation": is_new
        }

    except Exception as e:
        print(f"process_question 异常: {str(e)}")
        traceback.print_exc()
        raise


#获取历史记录
async def get_conversation_history(
        db: AsyncSession,
        user_id: int,
        conversation_id: int,
):
    try:
        messages = await message.get_messages_by_conversation(db, conversation_id,user_id )

        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "sources": msg.sources,
                    "created_at": msg.created_at.isoformat()
                }
                for msg in messages
            ]
        }

    except Exception as e:
        print(f"get_conversation_history 异常: {str(e)}")
        traceback.print_exc()
        raise