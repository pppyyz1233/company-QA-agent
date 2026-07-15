# routers/upload.py
import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from crud import user as user_crud
from stock.db import get_db
from utils.document_service import handle_document

router = APIRouter(prefix="/api/upload", tags=["文档上传"])
security = HTTPBearer()

# 上传目录
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/file")
async def upload_file(
        file: UploadFile = File(...),
        credentials: HTTPAuthorizationCredentials = Security(security),
        db: AsyncSession = Depends(get_db)
):
    """上传 PDF 或 Word 文件到知识库"""

    #验证用户并获取当前用户信息
    token = credentials.credentials
    current_user = await user_crud.get_user_by_token(db, token)

    if not current_user:
        raise HTTPException(status_code=401, detail="无效的令牌")

    #获取用户名作为上传者
    operator = current_user.username
    print(f"用户 {operator} 正在上传文件")

    #检查文件类型
    filename = file.filename
    if not (filename.endswith('.pdf') or filename.endswith('.docx')):
        raise HTTPException(status_code=400, detail="只支持 PDF 和 Word 文件")

    # 保存文件到本地
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    #根据文件类型解析
    try:
        if filename.endswith('.pdf'):
            result = pdf_service(file_path, filename, operator)
        else:
            result = word_service(file_path, filename, operator)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档解析失败: {str(e)}")

    return {
        "code": 200,
        "message": result,
        "data": {
            "filename": filename,
            "operator": operator,
            "status": result
        }
    }