"""PDF 텍스트 추출. 업로드 파일은 메모리(bytes)에서만 처리하고 디스크에 저장하지 않는다."""

import io

import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# 페이지당 평균 글자 수가 이 값보다 적으면 이미지 기반(스캔) PDF로 추정
MIN_CHARS_PER_PAGE = 20


class PDFExtractError(Exception):
    """사용자에게 그대로 보여줄 수 있는 PDF 처리 오류."""


@st.cache_data(show_spinner=False, max_entries=32, ttl=3600)  # Cloud 메모리 보호
def extract_pages(file_bytes: bytes) -> list[str]:
    """PDF bytes에서 페이지별 텍스트를 추출한다. 캐시 키는 파일 bytes."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except (PdfReadError, ValueError, OSError) as e:
        raise PDFExtractError("PDF 파일을 열 수 없습니다. 파일이 손상되었을 수 있습니다.") from e

    if reader.is_encrypted:
        # 열기 암호 없이 권한만 잠긴 PDF는 빈 암호로 열린다
        try:
            ok = reader.decrypt("")
        except Exception as e:
            raise PDFExtractError("암호화된 PDF는 열 수 없습니다.") from e
        if not ok:
            raise PDFExtractError("암호로 보호된 PDF입니다. 암호를 해제한 뒤 업로드해 주세요.")

    pages: list[str] = []
    try:
        for page in reader.pages:
            pages.append((page.extract_text() or "").strip())
    except Exception as e:
        raise PDFExtractError("PDF 텍스트를 읽는 중 오류가 발생했습니다. 파일이 손상되었을 수 있습니다.") from e
    return pages


def is_low_text(pages: list[str]) -> bool:
    """추출된 텍스트가 거의 없으면 True (이미지 기반 PDF 추정)."""
    if not pages:
        return True
    total = sum(len(p) for p in pages)
    return total / len(pages) < MIN_CHARS_PER_PAGE


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"
