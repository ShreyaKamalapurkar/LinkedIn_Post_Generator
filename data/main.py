import streamlit as st
from few_shot import FewShotPosts
from post_generator import generate_post

st.set_page_config(page_title="LinkedIn Post Generator", page_icon="📝", layout="centered")

# Cache FewShotPosts loader
@st.cache_resource
def get_fs():
    return FewShotPosts()

# Cache tags (no need to hash the fs object itself)
@st.cache_data
def get_available_tags():
    fs = get_fs()
    tags = fs.get_tags() or set()
    return sorted(tags) if tags else ["Productivity", "Mindset"]

def main():
    st.title("LinkedIn Post Generator")

    with st.spinner("Loading dataset…"):
        available_tags = get_available_tags()

    with st.form("controls"):
        col1, col2, col3 = st.columns(3)

        with col1:
            title = st.selectbox("Title", available_tags, index=0)

        with col2:
            length = st.selectbox("Length", ["Short", "Medium", "Long"], index=1)

        with col3:
            language = st.selectbox("Language", ["English", "Hinglish"], index=0)

        submitted = st.form_submit_button("Generate")

    if submitted:
        with st.spinner("Generating post…"):
            try:
                post = generate_post(length, language, title)
                st.subheader("Generated Post")
                st.write(post)
            except Exception as e:
                st.error(f"⚠️ Error while generating: {e}")

if __name__ == "__main__":
    main()
