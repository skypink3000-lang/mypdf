"""설정 로딩: Streamlit Cloud(st.secrets) → 로컬(.env) 순으로 읽는다."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_MODEL = "gpt-6.1-terra"


def get_config(key: str, default: str | None = None) -> str | None:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # 로컬에 secrets.toml이 없는 경우
    value = os.getenv(key)
    return value if value else default


OPENAI_API_KEY = get_config("OPENAI_API_KEY")
OPENAI_MODEL = get_config("OPENAI_MODEL", DEFAULT_MODEL)
