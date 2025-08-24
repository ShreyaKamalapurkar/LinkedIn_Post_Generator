import streamlit as st
from post_generator import generate_post
from few_shot import FewShotPosts

def main():
    st.title("LinkedIn Post Generator")

    # Load tags dynamically from dataset
    fs = FewShotPosts()
    available_tags = sorted(fs.get_tags()) if fs.get_tags() else ["Productivity", "Mindset"]

    col1, col2, col3 = st.columns(3)

    with col1:
        title = st.selectbox("Title", available_tags)

    with col2:
        length = st.selectbox("Length", ["Short", "Medium", "Long"])

    with col3:
        language = st.selectbox("Language", ["English", "Hinglish"])

    if st.button("Generate"):
        post = generate_post(length, language, title)
        st.subheader("Generated Post")
        st.write(post)

if __name__ == "__main__":
    main()
