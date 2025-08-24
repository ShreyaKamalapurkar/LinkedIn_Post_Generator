from llm_helper import llm
from few_shot import FewShotPosts

def get_length_str(length):
    if length == "Short":
        return "1 to 5 lines"
    if length == "Medium":
        return "6 to 10 lines"
    if length == "Long":
        return "11 to 15 lines"

def get_prompt(length, language, tag):
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

    # ---- main attempt (ignore language to avoid over-filtering) ----
    examples = few_shot.get_filtered_posts(length, None, tag)

    # ---- fallback 1: try by tag only (any length) ----
    if len(examples) == 0:
        examples = few_shot.get_filtered_posts(None, None, tag)

    # ---- fallback 2: try by length only ----
    if len(examples) == 0:
        examples = few_shot.get_filtered_posts(length, None)

    if len(examples) > 0:
        prompt += "\n4) Use the writing style as per the following example"
        # ✅ Only take the first example
        for i, post in enumerate(examples.to_dict("records")):
            post_text = post['text']
            prompt += f"\n\n Example{i+1} \n\n {post_text}"
            break   # stop after first example

    return prompt

def generate_post(length, language, tag, enforce_length=True):
    prompt = get_prompt(length, language, tag)
    response = llm.invoke(prompt)
    output = response.content.strip()

    # ✅ enforce line length trimming if model writes too long
    if enforce_length:
        length_map = {
            "Short":  (1, 5),    # between 1 and 5 lines
            "Medium": (6, 10),   # between 6 and 10 lines
            "Long":   (11, 15)   # between 11 and 15 lines
        }
        min_lines, max_lines = length_map.get(length, (None, None))

        if max_lines:
            lines = [line for line in output.splitlines() if line.strip()]  # ignore blank lines

            # ✅ clip to max_lines
            if len(lines) > max_lines:
                lines = lines[:max_lines]

            output = "\n".join(lines)

    # ✅ safety net: remove common preambles if they sneak in
    for bad_start in ["Here's", "This is", "Below is", "Here is"]:
        if output.startswith(bad_start):
            output = "\n".join(output.splitlines()[1:]).strip()
            break

    return output

if __name__ == "__main__":
   post = generate_post("Long", "English", "Productivity")
   print(post)
