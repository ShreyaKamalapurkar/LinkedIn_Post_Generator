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

import json
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# import LLM
from llm_helper import llm

#  Paths 
BASE_DIR = Path(__file__).resolve().parent         
RAW_PATH = BASE_DIR / "raw_posts.json"              
PROCESSED_PATH = BASE_DIR / "processed_posts.json"  


def process_post(raw_file_path: Path = RAW_PATH,
                 processed_file_path: Path = PROCESSED_PATH) -> None:
    enriched_posts = []

    # enrich posts with metadata
    with open(raw_file_path, encoding="utf-8") as file:
        posts = json.load(file)
        for post in posts:
            metadata = extract_metadata(post["text"])
            post_with_metadata = post | metadata
            enriched_posts.append(post_with_metadata)

    #  unify tags 
    canonical_tags = [
        "Entrepreneurship",
        "Business",
        "Leadership",
        "Motivation",
        "Finance",
        "Productivity",
        "Personal Growth",
        "Mindset",
    ]
    tag_mapping = get_unified_tag_mapping(enriched_posts, canonical_tags)
    enriched_posts = apply_unified_tags(enriched_posts, tag_mapping)

    #  processed output
    with open(processed_file_path, "w", encoding="utf-8") as f:
        json.dump(enriched_posts, f, ensure_ascii=False, indent=2)

    # print preview
    for epost in enriched_posts:
        print(epost)


def extract_metadata(post: str) -> dict:
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble.
    2. JSON object should have exactly three keys: line_count, language and tags.
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English or Hinglish (Hinglish means Hindi + English).
    5. For tags, pick from themes like: entrepreneurship, business, leadership, motivation, finance, productivity, personal growth, mindset.

    Here is the actual post on which you need to perform this task:
    {post}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"post": post})

    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        
        raise OutputParserException("Context too big. Unable to parse jobs.")

    return res


def get_unified_tag_mapping(posts_with_metadata: list, canonical_tags: list) -> dict:
    """
    Ask the LLM to map every observed (original) tag to one canonical tag.
    Returns a dict: {original_tag: unified_canonical_tag}
    """
    # collect unique tags
    unique_tags = set()
    for post in posts_with_metadata:
        for t in post.get("tags", []):
            if t is not None:
                unique_tags.add(str(t).strip())

    unique_tags_list = ", ".join(sorted(unique_tags))
    canonical_list = ", ".join(canonical_tags)

    # prompt 
    template = '''
I will give you a list of tags. You need to unify tags with the following requirements.

1. Tags are unified and merged to create a shorter list.
   Example 1: "Jobseekers", "Job Hunting" -> "Job Search"
   Example 2: "Motivation", "Inspiration", "Drive" -> "Motivation"
   Example 3: "Personal Growth", "Personal Development", "Self Improvement" -> "Personal Growth"
   Example 4: "Scam Alert", "Job Scam" -> "Scams"

2. Each unified tag MUST be one of these allowed (Title Case) categories:
   {canonical}

3. Output should be a JSON object. No preamble.
4. Output must map the ORIGINAL tag to the UNIFIED tag.
   For example: {{"Jobseekers": "Job Search", "Drive": "Motivation"}}

Here is the list of tags to unify:
{tags}
'''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(
        input={
            "tags": unique_tags_list,
            "canonical": canonical_list,
        }
    )

    try:
        json_parser = JsonOutputParser()
        mapping = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Failed to parse tag mapping from LLM response.")

    # sanitize mapping
    cleaned = {}
    for k, v in mapping.items():
        k_clean = str(k).strip()
        v_clean = str(v).strip()
        if k_clean and v_clean:
            cleaned[k_clean] = v_clean
    return cleaned


def apply_unified_tags(posts_with_metadata: list, mapping: dict) -> list:
    """
    Replace each post's tags using the LLM-provided mapping.
    De-duplicate per post while preserving order.
    """
    out = []
    for post in posts_with_metadata:
        orig_tags = post.get("tags", [])
        mapped = [mapping.get(t, t) for t in orig_tags]
        seen = set()
        deduped = [t for t in mapped if not (t in seen or seen.add(t))]
        out.append({**post, "tags": deduped})
    return out


if __name__ == "__main__":
    process_post()
