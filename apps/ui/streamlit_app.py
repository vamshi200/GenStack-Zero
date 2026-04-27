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


def init_chat_state() -> None:
    """Create chat history storage the first time the app loads."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_chat() -> None:
    """Remove all messages from the current chat session."""
    st.session_state.messages = []


def render_assistant_message(message: dict) -> None:
    """Render one assistant message with answer, context, and chunks."""
    st.markdown("**Answer**")
    st.write(message["answer"])

    answer_source = message.get("answer_source")
    if answer_source == "knowledge_base":
        st.caption("Source: Generative AI knowledge base")
    elif answer_source == "not_enough_context" and message.get("used_context"):
        st.caption("Source: Knowledge base had limited relevant context")
    elif answer_source == "not_enough_context":
        st.caption("Source: Outside the supported Generative AI domain")

    with st.expander("Used Context", expanded=False):
        used_context = message.get("used_context", "")
        if used_context:
            st.write(used_context)
        else:
            st.info("No context was used for this response.")

    with st.expander("Retrieved Chunks", expanded=False):
        chunks = message.get("chunks", [])
        if not chunks:
            st.info("No relevant chunks were returned.")
        else:
            for index, chunk in enumerate(chunks, start=1):
                st.markdown(f"**Chunk {index}**")
                st.write(chunk)


st.set_page_config(page_title="GenStack Zero", page_icon=":sparkles:", layout="centered")
init_chat_state()

header_left, header_right = st.columns([6, 1])

with header_left:
    st.title("GenStack Zero")
    st.caption("Zero-cost RAG and LLM GenAI system")

with header_right:
    st.button("Clear Chat", on_click=clear_chat, use_container_width=True)

st.caption("This assistant answers only Generative AI related questions using a fixed internal knowledge base.")

# Replay the existing conversation so the UI feels like a chat app.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            render_assistant_message(message)

# Keep the question input at the bottom of the page.
question = st.chat_input("Ask a question about the knowledge base")

if question:
    clean_question = question.strip()

    if not clean_question:
        st.warning("Please enter a question first.")
    else:
        user_message = {"role": "user", "content": clean_question}
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.write(clean_question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = ask_api(clean_question)

                assistant_message = {
                    "role": "assistant",
                    "answer": result.get("answer", "No answer returned."),
                    "answer_source": result.get("answer_source", ""),
                    "used_context": result.get("used_context", ""),
                    "chunks": result.get("chunks", []),
                }
                st.session_state.messages.append(assistant_message)
                render_assistant_message(assistant_message)
            except requests.exceptions.RequestException as error:
                error_message = {
                    "role": "assistant",
                    "answer": (
                        "Could not reach the FastAPI server. "
                        "Make sure the API is running on http://127.0.0.1:8000."
                    ),
                    "answer_source": "",
                    "used_context": "",
                    "chunks": [],
                }
                st.session_state.messages.append(error_message)
                render_assistant_message(error_message)
                st.code(str(error))
