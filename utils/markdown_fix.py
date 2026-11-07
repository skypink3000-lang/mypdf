"""한국어 마크다운 굵게(**) 렌더링 보정.

CommonMark 규칙상 닫는 `**` 앞이 문장부호이고 뒤가 글자(예: 조사)이면 강조로 인식되지 않는다.
  예) **결제 금액의 50%**를  →  `**`가 그대로 노출
  예) **「연차휴가 규정」**은 →  `**`가 그대로 노출
여는 `**`도 마찬가지(앞이 글자, 뒤가 문장부호). 이때 `**`와 문장부호 사이에 폭 없는 공백(U+200B)을
넣으면 구분자 판정 조건을 만족해 정상적으로 굵게 표시된다. 화면에는 보이지 않는다.
"""

import re
import unicodedata

ZWSP = "​"
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def _is_word(ch: str) -> bool:
    """공백도 문장부호도 아닌 문자 (CommonMark 구분자 판정 기준)."""
    return not ch.isspace() and not _is_punct(ch)


def fix_korean_bold(text: str) -> str:
    if "**" not in text:
        return text

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        before = text[m.start() - 1] if m.start() > 0 else " "
        after = text[m.end()] if m.end() < len(text) else " "
        opening = "**" + ZWSP if _is_punct(inner[0]) and _is_word(before) else "**"
        closing = ZWSP + "**" if _is_punct(inner[-1]) and _is_word(after) else "**"
        return f"{opening}{inner}{closing}"

    return _BOLD.sub(repl, text)
