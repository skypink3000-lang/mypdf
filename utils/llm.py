"""OpenAI 호출 및 프롬프트 관리.

- 모든 분석 함수는 문자열 조각을 내보내는 제너레이터를 반환한다 → st.write_stream으로 출력.
- 긴 문서는 청크로 나눠 부분 분석(map) 후 종합 분석(reduce)한다.
- OpenAI 예외는 LLMError(사용자용 한국어 메시지)로 변환된다.
"""

import re
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import streamlit as st
from openai import OpenAI

# ── 길이 설정 (문자 수 기준) ─────────────────────────────────
MAX_SINGLE_PASS_CHARS = 60_000   # 이보다 긴 문서는 map-reduce로 처리
CHUNK_CHARS = 20_000             # map 단계 청크 크기
QA_CONTEXT_CHARS = 60_000        # Q&A에 넣을 최대 컨텍스트 길이
QA_CHUNK_CHARS = 3_000           # Q&A 검색용 청크 크기
MAP_WORKERS = 4                  # map 단계 동시 요청 수 (요청 한도를 고려해 작게 유지)

ANALYSIS_TYPES = ["요약", "상세 분석", "키워드 추출", "비교 분석"]
LANGUAGES = {"한국어": "Korean", "English": "English"}

ProgressFn = Callable[[int, int], None] | None

# temperature를 지원하지 않는 것으로 확인된 모델 (예: gpt-6.1-terra는 기본값 1만 허용)
_NO_TEMPERATURE_MODELS: set[str] = set()


class LLMError(Exception):
    """사용자에게 그대로 보여줄 수 있는 LLM 호출 오류."""


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=120, max_retries=2)


def supports_temperature(model: str) -> bool:
    return model not in _NO_TEMPERATURE_MODELS


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _probe_temperature(_client: OpenAI, model: str) -> bool:
    """짧은 요청으로 temperature 지원 여부를 확인한다 (모델당 하루 1회 캐시).

    temperature 외의 오류는 예외로 올려 캐시되지 않게 한다.
    """
    try:
        _client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hi"}],
            temperature=0.5, max_completion_tokens=16,
        )
        return True
    except Exception as e:
        if _is_temperature_error(e):
            return False
        raise


def detect_temperature_support(client: OpenAI, model: str) -> bool:
    """앱 시작 시 사이드바 안내용. 확인할 수 없으면(네트워크 오류 등) 지원한다고 가정한다."""
    if model in _NO_TEMPERATURE_MODELS:
        return False
    try:
        supported = _probe_temperature(client, model)
    except Exception:
        return True
    if not supported:
        _NO_TEMPERATURE_MODELS.add(model)
    return supported


# ── OpenAI 호출 공통 ─────────────────────────────────────────
def _to_llm_error(e: Exception) -> LLMError:
    if isinstance(e, openai.AuthenticationError):
        return LLMError("API 키 인증에 실패했습니다. OPENAI_API_KEY 값을 확인해 주세요.")
    if isinstance(e, openai.PermissionDeniedError):
        return LLMError("이 API 키로는 해당 모델을 사용할 권한이 없습니다.")
    if isinstance(e, openai.NotFoundError):
        return LLMError("모델을 찾을 수 없습니다. OPENAI_MODEL 설정을 확인해 주세요.")
    if isinstance(e, openai.RateLimitError):
        return LLMError("요청 한도를 초과했거나 크레딧이 부족합니다. 잠시 후 다시 시도해 주세요.")
    if isinstance(e, openai.APITimeoutError):
        return LLMError("응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.")
    if isinstance(e, openai.APIConnectionError):
        return LLMError("OpenAI 서버에 연결할 수 없습니다. 네트워크를 확인해 주세요.")
    if isinstance(e, openai.BadRequestError):
        if "context" in str(e).lower() or "too long" in str(e).lower():
            return LLMError("문서가 너무 길어 처리할 수 없습니다. 파일 수나 분량을 줄여 주세요.")
        return LLMError("요청이 거부되었습니다(잘못된 요청). 잠시 후 다시 시도해 주세요.")
    if isinstance(e, openai.APIStatusError):
        return LLMError(f"OpenAI 서버 오류가 발생했습니다 (코드 {e.status_code}). 잠시 후 다시 시도해 주세요.")
    return LLMError("AI 분석 중 알 수 없는 오류가 발생했습니다.")


def _is_temperature_error(e: Exception) -> bool:
    return isinstance(e, openai.BadRequestError) and "temperature" in str(e).lower()


def _create(client: OpenAI, model: str, messages: list[dict], temperature: float, stream: bool):
    """temperature를 거부하는 모델이면 기본값으로 재시도하고 기억해 둔다."""
    kwargs = {"model": model, "messages": messages, "stream": stream}
    if supports_temperature(model):
        try:
            return client.chat.completions.create(**kwargs, temperature=temperature)
        except Exception as e:
            if not _is_temperature_error(e):
                raise
            _NO_TEMPERATURE_MODELS.add(model)
    return client.chat.completions.create(**kwargs)


def _complete(client: OpenAI, model: str, messages: list[dict], temperature: float) -> str:
    try:
        resp = _create(client, model, messages, temperature, stream=False)
    except Exception as e:
        raise _to_llm_error(e) from e
    return resp.choices[0].message.content or ""


def _stream(client: OpenAI, model: str, messages: list[dict], temperature: float) -> Iterator[str]:
    try:
        for chunk in _create(client, model, messages, temperature, stream=True):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except LLMError:
        raise
    except Exception as e:
        raise _to_llm_error(e) from e


# ── 텍스트 준비 ──────────────────────────────────────────────
def format_document(name: str, pages: list[str]) -> str:
    """페이지마다 [파일명 p.N] 표식을 붙인 문서 텍스트."""
    return "\n\n".join(f"[{name} p.{i}]\n{text}" for i, text in enumerate(pages, start=1) if text)


def _page_blocks(name: str, pages: list[str]) -> list[str]:
    return [f"[{name} p.{i}]\n{text}" for i, text in enumerate(pages, start=1) if text]


def split_chunks(blocks: list[str], size: int) -> list[str]:
    """페이지 블록을 size 이하 청크로 묶는다. 한 페이지가 size보다 길면 잘라서 나눈다."""
    chunks: list[str] = []
    current = ""
    for block in blocks:
        while len(block) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(block[:size])
            block = block[size:]
        if current and len(current) + len(block) + 2 > size:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{block}" if current else block
    if current:
        chunks.append(current)
    return chunks


# ── 프롬프트 ─────────────────────────────────────────────────
SYSTEM_ANALYST = (
    "You are an expert document analyst. Base everything strictly on the provided document text. "
    "Do not invent facts. The text contains page markers like [filename p.N]; cite them when useful. "
    "Format the answer in Markdown."
)

TASK_INSTRUCTIONS = {
    "요약": (
        "Summarize the document.\n"
        "1. An overall summary (one or two paragraphs).\n"
        "2. About 5 key points as a bulleted list."
    ),
    "상세 분석": (
        "Analyze the document in detail with these sections:\n"
        "1. Purpose of the document\n2. Structure / organization\n"
        "3. Main claims, findings and important data (with numbers where available)\n"
        "4. Implications and noteworthy points (risks, conditions, limitations)"
    ),
    "키워드 추출": (
        "Extract from the document:\n"
        "1. Core keywords (10-15) with a one-line explanation each\n"
        "2. Key entities grouped as: people, organizations, dates, numbers/amounts — each with brief context"
    ),
}

MAP_INSTRUCTION = (
    "This is one part of a longer document. Extract concise but complete notes from this part that will "
    "later be combined for the following task. Keep important facts, numbers, names, dates and the "
    "[filename p.N] markers.\n\nTask: {task}"
)


def _lang_line(language: str) -> str:
    return f"Write the answer in {LANGUAGES.get(language, language)}."


def _reduce_to_notes(
    client: OpenAI, model: str, name: str, pages: list[str], task: str, temperature: float,
    on_progress: ProgressFn,
) -> str:
    """긴 문서를 청크별 노트로 압축한다 (map 단계, 청크를 병렬 요청).

    워커 스레드는 OpenAI 호출만 하고, 진행률(Streamlit UI) 갱신은 호출 스레드에서만 한다.
    """
    chunks = split_chunks(_page_blocks(name, pages), CHUNK_CHARS)
    total = len(chunks)

    def work(chunk: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_ANALYST},
            {"role": "user", "content": f"{MAP_INSTRUCTION.format(task=task)}\n\n---\n{chunk}"},
        ]
        return _complete(client, model, messages, temperature)

    if on_progress:
        on_progress(0, total)
    notes: list[str] = [""] * total
    executor = ThreadPoolExecutor(max_workers=min(MAP_WORKERS, total))
    try:
        futures = {executor.submit(work, chunk): idx for idx, chunk in enumerate(chunks)}
        for done, future in enumerate(as_completed(futures), start=1):
            notes[futures[future]] = future.result()  # LLMError는 그대로 전파
            if on_progress:
                on_progress(done, total)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return "\n\n".join(f"(Part {i} notes)\n{n}" for i, n in enumerate(notes, start=1))


def _prepare_text(
    client: OpenAI, model: str, name: str, pages: list[str], task: str, temperature: float,
    on_progress: ProgressFn,
) -> tuple[str, bool]:
    """(분석할 텍스트, 노트로 압축했는지 여부)"""
    text = format_document(name, pages)
    if len(text) <= MAX_SINGLE_PASS_CHARS:
        return text, False
    return _reduce_to_notes(client, model, name, pages, task, temperature, on_progress), True


# ── 분석 함수 ────────────────────────────────────────────────
def _single_doc_task(
    kind: str, client: OpenAI, model: str, name: str, pages: list[str], language: str,
    temperature: float, on_progress: ProgressFn = None,
) -> Iterator[str]:
    task = TASK_INSTRUCTIONS[kind]
    text, condensed = _prepare_text(client, model, name, pages, task, temperature, on_progress)
    source = "notes extracted from each part of a long document" if condensed else "full document text"
    messages = [
        {"role": "system", "content": SYSTEM_ANALYST},
        {"role": "user", "content": f"{task}\n{_lang_line(language)}\n\nDocument: {name}\n"
                                    f"Below is the {source}.\n---\n{text}"},
    ]
    yield from _stream(client, model, messages, temperature)


def summarize(client, model, name, pages, language, temperature, on_progress=None) -> Iterator[str]:
    """전체 요약 + 핵심 포인트."""
    return _single_doc_task("요약", client, model, name, pages, language, temperature, on_progress)


def analyze(client, model, name, pages, language, temperature, on_progress=None) -> Iterator[str]:
    """상세 분석: 목적, 구조, 주요 주장/데이터, 시사점."""
    return _single_doc_task("상세 분석", client, model, name, pages, language, temperature, on_progress)


def extract_keywords(client, model, name, pages, language, temperature, on_progress=None) -> Iterator[str]:
    """핵심 키워드 및 주요 개체 추출."""
    return _single_doc_task("키워드 추출", client, model, name, pages, language, temperature, on_progress)


def compare(
    client: OpenAI, model: str, docs: dict[str, list[str]], language: str, temperature: float,
    on_progress: ProgressFn = None,
) -> Iterator[str]:
    """여러 문서 간 공통점/차이점 비교. 전체가 길면 문서별 노트로 압축 후 비교."""
    task = "Compare documents: topic, purpose, key claims/data, conclusions."
    full = {name: format_document(name, pages) for name, pages in docs.items()}
    if sum(len(t) for t in full.values()) <= MAX_SINGLE_PASS_CHARS:
        texts = full
    else:
        texts = {}
        for name, pages in docs.items():
            if len(full[name]) <= MAX_SINGLE_PASS_CHARS // len(docs):
                texts[name] = full[name]
            else:
                texts[name] = _reduce_to_notes(client, model, name, pages, task, temperature, on_progress)

    body = "\n\n".join(f"=== Document: {name} ===\n{text}" for name, text in texts.items())
    instruction = (
        f"Compare the following {len(texts)} documents.\n"
        "1. One-paragraph overview of each document\n"
        "2. Commonalities\n3. Differences (a Markdown table is welcome)\n"
        "4. Overall insight: how the documents relate or complement each other\n"
        f"{_lang_line(language)}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_ANALYST},
        {"role": "user", "content": f"{instruction}\n\n{body}"},
    ]
    yield from _stream(client, model, messages, temperature)


# ── Q&A ──────────────────────────────────────────────────────
SYSTEM_QA = (
    "You answer questions using ONLY the provided document excerpts. "
    "Each excerpt starts with a source marker like [filename p.N]. "
    "Cite the sources you used at the end of the answer, e.g. (출처: report.pdf p.3). "
    "If the answer is not in the documents, reply exactly that it cannot be found in the documents "
    "(in Korean: '문서에서 찾을 수 없습니다.'). "
    "Exception: questions about this conversation itself (e.g. what was asked or answered earlier) "
    "may be answered from the chat history without citing sources. "
    "Answer in the same language as the question. Use Markdown."
)


def _bigrams(text: str) -> set[str]:
    text = re.sub(r"\s+", "", text.lower())
    return {text[i:i + 2] for i in range(len(text) - 1)}


def build_qa_context(docs: dict[str, list[str]], question: str) -> str:
    """전체가 짧으면 전부, 길면 질문과 겹치는 문자 bigram이 많은 청크를 골라 컨텍스트를 만든다."""
    blocks = [b for name, pages in docs.items() for b in _page_blocks(name, pages)]
    full = "\n\n".join(blocks)
    if len(full) <= QA_CONTEXT_CHARS:
        return full

    chunks = split_chunks(blocks, QA_CHUNK_CHARS)
    q = _bigrams(question)
    ranked = sorted(range(len(chunks)), key=lambda i: len(q & _bigrams(chunks[i])), reverse=True)
    selected, total = [], 0
    for i in ranked:
        if total + len(chunks[i]) > QA_CONTEXT_CHARS:
            break
        selected.append(i)
        total += len(chunks[i])
    return "\n\n".join(chunks[i] for i in sorted(selected))  # 원래 문서 순서 유지


def answer_question(
    client: OpenAI, model: str, docs: dict[str, list[str]], history: list[dict], question: str,
    temperature: float,
) -> Iterator[str]:
    """업로드 문서 기반 Q&A. history는 [{"role": "user"|"assistant", "content": str}, ...]."""
    context = build_qa_context(docs, question)
    messages = [
        {"role": "system", "content": f"{SYSTEM_QA}\n\n--- DOCUMENTS ---\n{context}"},
        *history[-10:],  # 최근 대화만 유지
        {"role": "user", "content": question},
    ]
    yield from _stream(client, model, messages, temperature)


def run_analysis(
    kind: str, client: OpenAI, model: str, docs: dict[str, list[str]], language: str,
    temperature: float, on_progress: Callable[[str, int, int], None] | None = None,
) -> Iterator[str]:
    """분석 유형에 따라 실행. 단일 문서 분석은 파일마다 제목을 붙여 이어서 출력한다.

    on_progress(파일명, 현재 청크, 전체 청크)
    """
    if kind == "비교 분석":
        if len(docs) < 2:
            raise LLMError("비교 분석은 PDF가 2개 이상 필요합니다.")
        cb = (lambda i, n: on_progress("전체 문서", i, n)) if on_progress else None
        yield from compare(client, model, docs, language, temperature, cb)
        return

    func = {"요약": summarize, "상세 분석": analyze, "키워드 추출": extract_keywords}[kind]
    for idx, (name, pages) in enumerate(docs.items()):
        if len(docs) > 1:
            yield f"{'' if idx == 0 else chr(10) * 2}## 📄 {name}\n\n"
        cb = (lambda i, n, _name=name: on_progress(_name, i, n)) if on_progress else None
        yield from func(client, model, name, pages, language, temperature, cb)
