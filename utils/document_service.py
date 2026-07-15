import subprocess
import tempfile
from pathlib import Path

from utils.config import MINERU_ENABLED
from utils.handle_text import kb_service


def handle_document(file_path: str, filename: str, operator: str = "admin") -> str:
    """PDF/Word/txt/md 统一用 MinerU 解析后上传"""
    suffix = Path(file_path).suffix.lower()

    # txt / md 直接读
    if suffix in {".txt", ".md"}:
        content = Path(file_path).read_text(encoding="utf-8")
        print(f"直接读取完成，{len(content)}字")
        return kb_service.upload_by_str(content, filename)

    # PDF / Word / 其他 → MinerU
    if not MINERU_ENABLED:
        return "MinerU未启用，无法解析"

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["mineru", "-p", file_path, "-o", tmpdir, "--source", "modelscope"]
        print(f"  执行: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if r.returncode != 0:
            print(f"  MinerU失败: {r.stderr[-300:]}")
            return f"解析失败: {filename}"

        md_files = list(Path(tmpdir).rglob("*.md"))
        if not md_files:
            return "未找到解析结果"

        content = md_files[0].read_text(encoding="utf-8")
        print(f"  MinerU解析完成，{len(content)}字")
        return kb_service.upload_by_str(content, filename)