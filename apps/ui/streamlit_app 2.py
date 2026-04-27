import requests
import streamlit as st


# Keep the API URL in one place so it is easy to change later.
API_URL = "http://127.0.0.1:8000/query"


def ask_api(question: str) -> dict:
    """Send the user's question to the FastAPI backend."""
    payload = {"question": question}

    # requests handles the HTTP call and returns the JSON response.
    response = requests.post(API_URL, json=payload, timeout=60)
    response.raise_for_status()

    return response.json()


st.set_page_config(page_title="GenStack Zero", page_icon=":sparkles:", layout="centered")

# Simple page heading for the demo UI.
st.title("GenStack Zero")
st.caption("Zero-cost RAG and LLM GenAI system")

# Let the user type a question in plain language.
question = st.text_input("Ask a question", placeholder="Example: What is QLoRA?")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            with st.spinner("Generating answer..."):
                result = ask_api(question.strip())

            st.subheader("Final Answer")
            st.write(result.get("answer", "No answer returned."))

            st.subheader("Used Context")
            used_context = result.get("used_context", "")
            st.text_area("Context sent to the model", value=used_context, height=180)

            st.subheader("Retrieved Chunks")
            chunks = result.get("chunks", [])

            if not chunks:
                st.info("No relevant chunks were returned.")
            else:
                for index, chunk in enumerate(chunks, start=1):
                    st.markdown(f"**Chunk {index}**")
                    st.write(chunk)
        except requests.exceptions.RequestException as error:
            st.error(
                "Could not reach the FastAPI server. "
                "Make sure the API is running on http://127.0.0.1:8000."
            )
            st.code(str(error))
