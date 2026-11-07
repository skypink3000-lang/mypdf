import hashlib
from datetime import datetime, timedelta, timezone

import streamlit as st

st.set_page_config(page_title="PDF 분석기", page_icon="📄", layout="wide")

from utils.config import OPENAI_API_KEY, OPENAI_MODEL  # noqa: E402
from utils.llm import (  # noqa: E402
    ANALYSIS_TYPES,
    LANGUAGES,
    LLMError,
    answer_question,
    detect_temperature_support,
    get_client,
    run_analysis,
)
from utils.markdown_fix import fix_korean_bold  # noqa: E402
from utils.pdf_reader import PDFExtractError, extract_pages, format_size, is_low_text  # noqa: E402

# 비용·메모리 보호용 제한 (파일당 크기 제한은 .streamlit/config.toml의 maxUploadSize)
MAX_FILES = 10
MAX_TOTAL_PAGES = 500
MAX_UPLOAD_MB = st.get_option("server.maxUploadSize")
KST = timezone(timedelta(hours=9))

# ── 스타일 (커스텀 CSS는 여기에만 둔다) ─────────────────────
CUSTOM_CSS = """
<style>
/* 넓은 화면에서 본문 폭 제한 → 가독성 */
.block-container { max-width: 1100px; padding-top: 2.5rem; }
/* AI 결과 안의 마크다운 제목이 페이지 제목보다 커지지 않도록 */
.st-key-result h1 { font-size: 1.6rem; }
.st-key-result h2 { font-size: 1.35rem; }
.st-key-result h3 { font-size: 1.15rem; }
/* 모바일: 좌우 여백 축소 */
@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.5rem; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── 세션 상태 ────────────────────────────────────────────────
DEFAULTS = {
    "doc_signature": None,
    "analysis": None,        # {"kind", "language", "text"}
    "analysis_error": None,  # 마지막 분석 오류 메시지
    "chat_history": [],      # [{"role", "content"}]
    "running": False,        # 분석 실행 중 (중복 클릭 방지)
    "pending_toast": None,   # rerun 후 보여줄 알림
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

if st.session_state.pending_toast:
    st.toast(st.session_state.pending_toast, icon="✅")
    st.session_state.pending_toast = None


def reset_chat() -> None:
    st.session_state.chat_history = []
    st.toast("대화를 초기화했습니다.", icon="🧹")


def start_analysis() -> None:
    st.session_state.running = True
    st.session_state.analysis_error = None


busy = st.session_state.running

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 PDF 분석기")
    st.badge(OPENAI_MODEL, icon=":material/smart_toy:", color="violet")

    st.divider()
    st.markdown("**⚙️ 분석 설정**")
    analysis_kind = st.selectbox("분석 유형", ANALYSIS_TYPES, disabled=busy)
    language = st.radio("출력 언어", list(LANGUAGES), horizontal=True, disabled=busy)
    # 앱 시작 시 한 번 확인(하루 캐시) → 첫 분석 전에도 안내가 정확하다
    temp_ok = not OPENAI_API_KEY or detect_temperature_support(get_client(OPENAI_API_KEY), OPENAI_MODEL)
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.3, 0.1, disabled=busy or not temp_ok,
        help="낮을수록 일관된 답변, 높을수록 다양한 답변" if temp_ok else "현재 모델은 조정을 지원하지 않습니다.",
    )
    if not temp_ok:
        st.caption("현재 모델은 temperature 조정을 지원하지 않아 기본값(1)으로 실행됩니다.")

    st.divider()
    st.markdown("**💬 대화**")
    st.button("대화 초기화", on_click=reset_chat, width="stretch", disabled=busy)

# ── 헤더 · 업로드 ────────────────────────────────────────────
st.title("📄 PDF 분석기")
st.caption("PDF를 업로드하면 AI가 내용을 요약·분석하고, 문서에 대한 질문에 답합니다.")

if not OPENAI_API_KEY:
    st.error(
        "OpenAI API 키가 설정되지 않았습니다.\n\n"
        "- **로컬**: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY=sk-...` 를 추가하세요.\n"
        "- **Streamlit Cloud**: 앱 설정 → Secrets에 `OPENAI_API_KEY = \"sk-...\"` 를 추가하세요."
    )
    st.stop()

client = get_client(OPENAI_API_KEY)

uploaded_files = st.file_uploader(
    "PDF 파일 업로드 (여러 개 선택 가능)",
    type=["pdf"],
    accept_multiple_files=True,
    disabled=busy,
    help=f"파일당 최대 {MAX_UPLOAD_MB}MB, 최대 {MAX_FILES}개, 전체 {MAX_TOTAL_PAGES}페이지까지 분석할 수 있습니다.",
)
st.caption(
    f"권장: 파일당 {MAX_UPLOAD_MB}MB 이하 · 최대 {MAX_FILES}개 · 전체 {MAX_TOTAL_PAGES}페이지 이하. "
    "텍스트가 있는 PDF만 분석할 수 있습니다(스캔 이미지 PDF 불가)."
)

# 업로드 파일 구성이 바뀌면 이전 분석 결과·대화 이력 초기화
signature = tuple((f.name, hashlib.md5(f.getvalue()).hexdigest()) for f in (uploaded_files or []))
if signature != st.session_state.doc_signature:
    st.session_state.doc_signature = signature
    st.session_state.analysis = None
    st.session_state.analysis_error = None
    st.session_state.chat_history = []

# ── 빈 상태 안내 ─────────────────────────────────────────────
if not uploaded_files:
    st.markdown("#### 이렇게 사용하세요")
    steps = [
        ("📤", "1. 업로드", "분석할 PDF 파일을 위 영역에 끌어다 놓거나 선택하세요. 여러 개도 가능합니다."),
        ("📊", "2. 분석", "사이드바에서 분석 유형과 언어를 고른 뒤 **분석 시작**을 누르세요."),
        ("💬", "3. 질문", "**Q&A** 탭에서 문서 내용을 질문하세요. 근거 페이지도 함께 알려 드립니다."),
    ]
    for col, (icon, title, desc) in zip(st.columns(3), steps):
        with col.container(border=True, height="stretch"):
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)
    st.stop()

# ── 파일 읽기 ────────────────────────────────────────────────
def unique_name(name: str, taken: set[str]) -> str:
    """같은 이름의 파일이 여러 개면 '이름 (2).pdf' 형태로 구분한다 (dict 키 덮어쓰기 방지)."""
    if name not in taken:
        return name
    stem, dot, ext = name.rpartition(".")
    stem, ext = (stem, f".{ext}") if dot else (name, "")
    n = 2
    while f"{stem} ({n}){ext}" in taken:
        n += 1
    return f"{stem} ({n}){ext}"


documents: dict[str, list[str]] = {}   # 텍스트가 있는 문서 (분석 대상)
all_pages: dict[str, list[str]] = {}   # 원문 탭용
if len(uploaded_files) > MAX_FILES:
    st.error(f"한 번에 최대 {MAX_FILES}개 파일까지 분석할 수 있습니다. 현재 {len(uploaded_files)}개가 업로드되었습니다. 일부 파일을 삭제해 주세요.")
    st.stop()

rows = []
file_errors: list[str] = []    # 읽을 수 없는 파일 (st.error)
file_warnings: list[str] = []  # 읽었지만 텍스트가 거의 없는 파일 (st.warning)
taken_names: set[str] = set()
with st.spinner("PDF에서 텍스트를 추출하는 중입니다..."):
    for file in uploaded_files:
        name = unique_name(file.name, taken_names)
        taken_names.add(name)
        file_bytes = file.getvalue()
        row = {"파일명": name, "페이지": "—", "크기": format_size(len(file_bytes))}
        try:
            pages = extract_pages(file_bytes)
        except PDFExtractError as e:
            rows.append({**row, "상태": "❌ 오류"})
            file_errors.append(f"**{name}** — {e}")
            continue
        all_pages[name] = pages
        if any(pages):
            documents[name] = pages
        row["페이지"] = str(len(pages))
        if is_low_text(pages):
            rows.append({**row, "상태": "⚠️ 텍스트 없음"})
            file_warnings.append(f"**{name}** — 텍스트를 추출할 수 없습니다(이미지 기반 PDF일 수 있음).")
        else:
            rows.append({**row, "상태": "✅ 정상"})

st.dataframe(
    rows,
    hide_index=True,
    width="stretch",
    column_order=["상태", "파일명", "페이지", "크기"],
    column_config={
        "상태": st.column_config.TextColumn(width="small"),
        "페이지": st.column_config.TextColumn(width="small"),
        "크기": st.column_config.TextColumn(width="small"),
    },
)
for msg in file_errors:
    st.error(msg)
for msg in file_warnings:
    st.warning(msg)

if not documents:
    st.warning("분석할 수 있는 텍스트가 있는 PDF가 없습니다. 다른 파일을 업로드해 주세요.")
    st.stop()

total_pages = sum(len(p) for p in documents.values())
if total_pages > MAX_TOTAL_PAGES:
    st.error(
        f"전체 {total_pages}페이지로 분석 가능한 분량({MAX_TOTAL_PAGES}페이지)을 넘었습니다. "
        "API 비용과 처리 시간을 위해 파일 수를 줄여 주세요."
    )
    st.stop()

tab_analysis, tab_qa, tab_raw = st.tabs(["📊 분석 결과", "💬 Q&A", "📄 원문"])

# ── 탭 1: 분석 결과 ─────────────────────────────────────────
with tab_analysis:
    compare_blocked = analysis_kind == "비교 분석" and len(documents) < 2

    col_info, col_btn = st.columns([3, 1], vertical_alignment="center")
    col_info.markdown(f"**{analysis_kind}** · {language} · 문서 {len(documents)}개")
    col_btn.button(
        "분석 시작",  # 라벨이 바뀌면 새 위젯으로 취급돼 버튼이 중복 표시되므로 고정
        type="primary",
        icon=":material/play_arrow:",
        on_click=start_analysis,
        disabled=busy or compare_blocked,
        width="stretch",
        help="비교 분석은 텍스트가 있는 PDF가 2개 이상 필요합니다." if compare_blocked else None,
    )
    if compare_blocked:
        st.caption("⚠️ 비교 분석은 텍스트가 있는 PDF가 2개 이상 필요합니다. 다른 분석 유형을 선택하거나 파일을 추가해 주세요.")

    if busy:
        status = st.status("분석 중입니다...", expanded=False)
        bar = status.empty()

        def on_progress(name: str, i: int, n: int) -> None:
            status.update(label=f"긴 문서 분석 중 — {name} ({i}/{n} 청크 완료)")
            bar.progress(i / n, text=f"{i}/{n} 청크")

        try:
            # 완료 후 st.rerun()으로 다시 그리므로 아래 결과 컨테이너와 key가 겹치지 않는다
            with st.container(border=True, key="result"):
                text = st.write_stream(
                    run_analysis(analysis_kind, client, OPENAI_MODEL, documents, language, temperature, on_progress)
                )
            if isinstance(text, str) and text.strip():
                st.session_state.analysis = {
                    "kind": analysis_kind,
                    "language": language,
                    "text": text,
                    "files": list(documents),
                    "created": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
                }
                st.session_state.pending_toast = "분석이 완료되었습니다."
            else:
                st.session_state.analysis = None
                st.session_state.analysis_error = "AI가 빈 응답을 반환했습니다. 다시 시도해 주세요."
        except LLMError as e:
            st.session_state.analysis = None
            st.session_state.analysis_error = str(e)
        finally:
            st.session_state.running = False
        st.rerun()

    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)

    result = st.session_state.analysis
    if result:
        body = fix_korean_bold(result["text"])
        with st.container(border=True, key="result"):
            st.caption(f"{result['kind']} · {result['language']} · {result['created']}")
            st.markdown(body)
        header = (
            f"# PDF 분석 결과 — {result['kind']}\n\n"
            f"- 대상 파일: {', '.join(result['files'])}\n"
            f"- 출력 언어: {result['language']}\n"
            f"- 모델: {OPENAI_MODEL}\n"
            f"- 생성 시각: {result['created']} (KST)\n\n---\n\n"
        )
        st.download_button(
            "결과 다운로드 (.md)",
            data=header + body,
            file_name=f"pdf_analysis_{result['kind'].replace(' ', '_')}.md",
            mime="text/markdown",
            icon=":material/download:",
            on_click="ignore",
        )
    elif not st.session_state.analysis_error:
        st.info("사이드바에서 분석 유형을 고른 뒤 **분석 시작**을 눌러 주세요.")

# ── 탭 2: Q&A ───────────────────────────────────────────────
with tab_qa:
    st.caption("업로드한 문서 내용에 대해 질문하세요. 답변 끝에 근거 파일명과 페이지가 표시됩니다.")
    chat_box = st.container()
    # 입력창은 대화 목록 아래에 두되, 질문 여부를 먼저 알아야 예시 문구 표시를 정할 수 있다
    question = st.chat_input("문서에 대해 질문하세요", disabled=busy)
    with chat_box:
        if not st.session_state.chat_history and not question:
            st.caption("예) “이 문서의 핵심 결론은 무엇인가요?” · “3페이지의 표를 설명해 주세요.”")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(fix_korean_bold(msg["content"]))

    if question:
        history = st.session_state.chat_history
        prior = list(history)
        # 답변 생성 중 새 질문이 들어오면 Streamlit이 이 실행을 중단(rerun)한다.
        # 질문과 답변 자리를 먼저 함께 저장해 두고(순서 고정), 결과는 그 자리를 제자리 수정한다.
        # 중단된 실행의 finally가 다음 실행보다 늦게 돌아도 대화 순서가 꼬이거나 사라지지 않는다.
        interrupted_note = "_(새 질문이 들어와 답변이 중단되었습니다.)_"
        user_msg = {"role": "user", "content": question}
        bot_msg = {"role": "assistant", "content": interrupted_note}
        history.extend([user_msg, bot_msg])
        parts: list[str] = []
        outcome = "interrupted"

        def collect():
            for piece in answer_question(client, OPENAI_MODEL, documents, prior, question, temperature):
                parts.append(piece)
                yield piece

        with chat_box:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    st.write_stream(collect())
                    outcome = "done"
                except LLMError as e:
                    outcome = "error"
                    st.error(str(e))
                finally:
                    if outcome == "done":
                        bot_msg["content"] = "".join(parts).strip() or "_(AI가 빈 응답을 반환했습니다. 다시 질문해 주세요.)_"
                    elif outcome == "interrupted":
                        partial = "".join(parts).strip()
                        if partial:
                            bot_msg["content"] = f"{partial}\n\n{interrupted_note}"
                    else:  # 오류: 질문·답변 자리를 이력에서 제거 (재시도 가능)
                        history[:] = [m for m in history if m is not user_msg and m is not bot_msg]
        if outcome == "done":
            st.rerun()  # 스트리밍 중 표시된 원문 대신 보정된 마크다운으로 다시 그린다

# ── 탭 3: 원문 ──────────────────────────────────────────────
with tab_raw:
    st.caption("PDF에서 추출한 텍스트입니다. AI 분석은 이 텍스트를 기반으로 합니다.")
    for name, pages in all_pages.items():
        with st.expander(f"{name} ({len(pages)}페이지)"):
            for i, text in enumerate(pages, start=1):
                st.markdown(f"**p.{i}**")
                st.text(text if text else "(텍스트 없음)")
