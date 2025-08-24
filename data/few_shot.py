import json
import pandas as pd

class FewShotPosts:
    def __init__(self, file_path="processed_posts.json"):
        self.df = None
        self.unique_tags = None
        self.load_posts(file_path)

    def load_posts(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            posts = json.load(f)
            self.df = pd.json_normalize(posts)

            # add length column
            self.df["length"] = self.df["line_count"].apply(self.categorize_length)

            # collect unique tags if present
            if "tags" in self.df.columns:
                all_tags = (
                    self.df["tags"]
                    .apply(lambda x: x if isinstance(x, list) else [])
                    .sum()
                )
                self.unique_tags = set(all_tags)
            else:
                print("⚠️ No 'tags' column found. Columns are:", self.df.columns)

    def categorize_length(self, line_count):
        if line_count < 5:
            return "Short"
        elif 5 <= line_count <= 15:
            return "Medium"
        else:
            return "Long"

    def get_tags(self):
        return self.unique_tags

    def get_filtered_posts(self, length=None, language=None, *tags, match="any"):
        """
        length: 'Short' | 'Medium' | 'Long' (optional)
        language: e.g. 'Hinglish' (optional)
        *tags: one or more tag strings (optional)
        match: 'any' (default) or 'all' to require all tags
        """
        df = self.df

        # filter by length
        if length is not None and "length" in df.columns:
            df = df[df["length"] == length]

        # filter by language
        if language is not None and "language" in df.columns:
            df = df[df["language"] == language]

        # filter by tags
        if tags and "tags" in df.columns:
            # normalize each cell to a list
            tag_series = df["tags"].apply(lambda x: x if isinstance(x, list) else [])
            if match == "all":
                mask = tag_series.apply(lambda lst: all(t in lst for t in tags))
            else:  # 'any'
                mask = tag_series.apply(lambda lst: any(t in lst for t in tags))
            df = df[mask]

        return df

if __name__ == "__main__":
    fs = FewShotPosts()

    # Example matching ANY of the tags:
    posts = fs.get_filtered_posts("Long", "English", "Personal Growth")
    print(posts)

    
