# Copyright 2025 Shreya Kamalapurkar
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# data/llm_helper.py
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq

def _key_from_streamlit() -> Optional[str]:
    """Read GROQ_API_KEY from Streamlit Cloud secrets if available."""
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None

def _key_from_env() -> Optional[str]:
    """Read GROQ_API_KEY from local .env (for dev)."""
    load_dotenv()  # no-op on cloud
    return os.getenv("GROQ_API_KEY")

def get_api_key() -> Optional[str]:
    return _key_from_streamlit() or _key_from_env()

def get_llm() -> ChatGroq:
    """
    Lazily create the Groq LLM. This avoids crashes during module import
    on Streamlit Cloud when the key isn't set yet.
    """
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. "
            "On Streamlit Cloud, set it in Settings → Secrets. "
            "Locally, create a .env with GROQ_API_KEY=your_key."
        )
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
    )

# Optional local quick test: `python data/llm_helper.py`
if __name__ == "__main__":
    try:
        llm = get_llm()
        resp = llm.invoke("Name two main ingredients in a samosa.")
        print(resp.content)
    except Exception as e:
        print("LLM test failed:", e)
