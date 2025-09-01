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

# data/few_shot.py
from pathlib import Path
import json
import re
import pandas as pd

DATA_DIR = Path(__file__).parent
PROCESSED = DATA_DIR / "processed_posts.json"
RAW = DATA_DIR / "raw_posts.json"

TOPIC_RULES = {
    "Productivity":  r"\b(work|outwork|grind|disciplin|consisten|habit|read)\w*",
    "Leadership":    r"\b(leader|leadership|team|captain|delegate|founder|startup|company)\b",
    "Mindset":       r"\b(mindset|belief|dream|values|respect|inferior|insecure|confidence)\b",
    "Work-Life":     r"\b(balance|burnt\s*out|burnout|relationships|guilt)\b",
    "Wealth":        r"\b(money|rich|wealth|paise)\b",
    "Psychology":    r"\b(psychology|behavio(u)?r|pattern)\b",
    "Control/Emotion": r"\b(emotion|react|control)\b",
    "Business":        r"\b(startup|amazon|instagram|feature|market|focus)\b",
}
DEFAULT_TOPIC = "General"

def _infer_topic(text: str) -> str:
    t = text.lower()
    for topic, pat in TOPIC_RULES.items():
        if re.search(pat, t):
            return topic
    return DEFAULT_TOPIC

def _count_lines(text: str) -> int:
    return max(1, text.count("\n") + 1)

class FewShotPosts:
    def __init__(self, file_path: str | Path = PROCESSED):
        self.df: pd.DataFrame | None = None
        self.unique_tags: set[str] | None = None
        self.load_posts(file_path)

    def _build_from_raw(self) -> pd.DataFrame:
        if not RAW.exists():
            return pd.DataFrame(columns=["text","engagement","line_count","length","language","tags","title"])
        raw = json.loads(RAW.read_text(encoding="utf-8"))
        rows = []
        for item in raw:
            text = (item.get("text") or "").strip()
            lc = _count_lines(text)
            topic = _infer_topic(text)
            rows.append({
                "text": text,
                "engagement": item.get("engagement"),
                "line_count": lc,
                "length": ("Short" if lc < 5 else "Medium" if lc <= 15 else "Long"),
                "language": "English",
                "tags": [topic],
                "title": topic,
            })
        df = pd.DataFrame(rows)
        # best-effort cache
        try:
            PROCESSED.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return df

    def load_posts(self, file_path: str | Path):
        file_path = Path(file_path)
        if file_path.exists():
            posts = json.loads(file_path.read_text(encoding="utf-8"))
            df = pd.json_normalize(posts)
        else:
            df = self._build_from_raw()

        if "line_count" not in df.columns:
            df["line_count"] = df["text"].fillna("").apply(_count_lines)
        if "length" not in df.columns:
            df["length"] = df["line_count"].apply(lambda n: "Short" if n < 5 else "Medium" if n <= 15 else "Long")
        if "language" not in df.columns:
            df["language"] = "English"
        if "tags" not in df.columns:
            if "title" in df.columns:
                df["tags"] = df["title"].apply(lambda v: [v] if pd.notna(v) else [])
            else:
                df["tags"] = df["text"].fillna("").apply(lambda t: [_infer_topic(t)] if t else [])

        self.df = df
        all_tags = df["tags"].apply(lambda x: x if isinstance(x, list) else []).sum()
        self.unique_tags = set(all_tags)

    def get_tags(self) -> set[str] | None:
        return self.unique_tags

    def get_filtered_posts(self, length=None, language=None, *tags, match="any") -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()
        df = self.df
        if length is not None:
            df = df[df["length"] == length]
        if language is not None:
            df = df[df["language"] == language]
        if tags:
            s = df["tags"].apply(lambda x: x if isinstance(x, list) else [])
            df = df[s.apply(lambda lst: all(t in lst for t in tags) if match=="all" else any(t in lst for t in tags))]
        return df
