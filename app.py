
import streamlit as st
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.output_parsers import StrOutputParser
parser=StrOutputParser()

@st.cache_resource
def get_transcript_api():
    return YouTubeTranscriptApi()

yt_api = get_transcript_api()

st.title("YouTube Video Summarizer")
st.subheader("Summarize YouTube videos using Groq LLM")
with st.sidebar:
    st.title("Configuration")
    api_key = st.text_input("Enter your Groq API Key", type="password")



prompt=ChatPromptTemplate.from_messages([
    ("system","""You are an intelligent assistant trained to summarize YouTube video transcripts.

Your goal is to create a clear, structured, and detailed summary that captures all important ideas, insights, and examples.
Avoid copying phrases directly from the transcript — use your own words for clarity and flow.

Follow this structure in your answer:
1. **Overview:** Briefly describe what the video is about and its main purpose.
2. **Main Points:** List the key ideas, arguments, or sections discussed in the video.
3. **Supporting Details:** Include important examples.
4. **Takeaways:** Highlight the main lessons, conclusions, or insights the viewer should remember.

Keep the summary concise and easy to read.
Ignore timestamps, filler words, or irrelevant parts of the transcript.
"""),
    ("user","{text}")
])


url= st.text_input("Enter YouTube Video URL",label_visibility="collapsed")




@st.cache_data(show_spinner=False)
def extract(url):
    try:
        video_id= url.split("=")[1]

        fetched = yt_api.fetch(video_id) 
        raw = fetched.to_raw_data()               
        full_yt_text = " ".join(d["text"] for d in raw)          
        return full_yt_text

    except Exception as e:
        return None

@st.cache_resource
def get_groq_model(api_key: str):
    """Cache LLM instance since model loading is expensive."""
    return ChatGroq(api_key=api_key, model="openai/gpt-oss-20b", temperature=0.6)


def summarize(text):
    try:
        llm = get_groq_model(api_key)
        chain = prompt | llm | parser
        response = chain.invoke({"text": text})
        return response
    except Exception as e:
        return st.error(f"Please enter a valid YouTube Video URL. {e}")





if st.button("Summarize"):
    if not api_key.strip() or not url.strip():
        st.error("Please enter your Groq API Key and YouTube Video URL.")
    elif "youtube" not in url:
        st.error("Please enter a valid YouTube Video URL.")
    else:
        with st.spinner("Extracting video content..."):
            yt_text = extract(url)

            if yt_text is None:
                st.error("Please enter a valid YouTube Video URL.")

            else:
                with st.spinner("Generating summary..."):
                    summary = summarize(yt_text)
                    if summary:
                        st.subheader("Video Thumbnail")
                        video_id = url.split("=")[1]
                        st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg", use_container_width=True)
                        st.subheader("Video Summary")
                        st.success(summary)