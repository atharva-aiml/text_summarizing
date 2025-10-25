import validators, streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit app

st.set_page_config(page_title="Summarize Text from Youtube or Website")
st.title("Text Summarizer")
st.subheader("Summarize URL")


## Get the Groq API KEY and url field to be summarized

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type = "password")



## Creating Prompt
prompt_template = """
Provide summary of the following content in 300 words:
Content:{text}
"""

prompt = PromptTemplate(template = prompt_template, input_variables=["text"])

## Creating output parser

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

## Combining docs

from langchain_core.runnables import RunnableLambda

def combine_docs(docs):
    # Handle tuple output (some YouTube videos return tuple)
    if isinstance(docs, tuple):
        docs_list = docs[0]
    else:
        docs_list = docs

    # Handle empty documents
    if not docs_list:
        return {"text": ""}

    # Join all page contents
    combined_text = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs_list)
    return {"text": combined_text}

combine_docs_runnable = RunnableLambda(combine_docs)



generic_url = st.text_input("URL", label_visibility="collapsed")

if st.button("Summarize the content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip:
        st.error("Please provide the information to get started")
        st.stop()
    elif not validators.url(generic_url):
        st.error("Please enter valid URL")
        st.stop()
    else:
        try:
            llm = ChatGroq(model = "llama-3.1-8b-instant", groq_api_key=groq_api_key)
            with st.spinner("Waiting..."):
                ### Loading the website data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info = True)
                else:
                    loader = UnstructuredURLLoader(urls = [generic_url], ssl_verify= False,
                                                   headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                
                docs = loader.load()

                ## Chain for summarization
                combine_docs_runnable = RunnableLambda(combine_docs)
                summarize_chain = combine_docs_runnable | prompt | llm | parser
                output_summary = summarize_chain.invoke(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception("Exception:{e}")