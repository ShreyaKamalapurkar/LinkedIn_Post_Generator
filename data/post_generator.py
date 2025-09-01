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

# data/post_generator.py
from __future__ import annotations

from llm_helper import get_llm
from few_shot import FewShotPosts

# Cache a single LLM instance per process (faster; avoids re-auth each call)
_LLM = None
def _model():
    global _LLM
    if _LLM is None:
        _LLM = get_llm()
    return _LLM

def get_length_str(length: str) -> str:
    if length == "Short":
        return "1 to 5 lines"
    if length == "Medium":
        return "6 to 10 lines"
    if length == "Long":
        return "11 to 15 lines"
    return "1 to 10 lines"

def get_prompt(length: str, language: str, tag: str) -> str:
    length_str = get_length_str(length)
    prompt = f"""
Generate a LinkedIn post using the below Information.

1) Topic: {tag}
2) Length: {length} ({length_str})
   - Short = 1 to 5 lines
   - Medium = 6 to 10 lines
   - Long = 11 to 15 lines
3) Language: {language}
   - If the Language is Hinglish it means it is mix of Hindi and English.
   - The script for the generated post should always be English.

⚠️ Important:
- Do NOT write any preamble like "Here’s a post", "This is a sample", "Below is...", etc.
- Directly output only the LinkedIn post content.
"""
    few_shot = FewShotPosts()

    # Main attempt (ignore language to avoid over-filtering)
    examples = few_shot.get_filtered_posts(length, None, tag)

    # Fallback 1: by tag only
    if len(examples) == 0:
        examples = few_shot.get_filtered_posts(None, None, tag)

    # Fallback 2: by length only
    if len(examples) == 0:
        examples = few_shot.get_filtered_posts(length, None)

    if len(examples) > 0:
        prompt += "\n4) Use the writing style as per the following example"
        # Only include the first example
        for i, post in enumerate(examples.to_dict("records")):
            post_text = post.get("text", "")
            if post_text:
                prompt += f"\n\n Example{i+1} \n\n {post_text}"
            break

    return prompt.strip()

def generate_post(length: str, language: str, tag: str, enforce_length: bool = True) -> str:
    prompt = get_prompt(length, language, tag)

    # Invoke LLM (lazy init, works with st.secrets or .env via llm_helper)
    llm = _model()
    response = llm.invoke(prompt)
    output = (response.content or "").strip()

    # Enforce length bounds by line count
    if enforce_length:
        length_map = {
            "Short":  (1, 5),
            "Medium": (6, 10),
            "Long":   (11, 15),
        }
        _, max_lines = length_map.get(length, (None, None))
        if max_lines:
            lines = [line for line in output.splitlines() if line.strip()]
            if len(lines) > max_lines:
                lines = lines[:max_lines]
            output = "\n".join(lines).strip()

    # Remove common preambles if they sneak in
    bad_starts = ("Here's", "This is", "Below is", "Here is")
    for bad in bad_starts:
        if output.startswith(bad):
            output = "\n".join(output.splitlines()[1:]).strip()
            break

    return output

if __name__ == "__main__":
    print(generate_post("Long", "English", "Productivity"))
