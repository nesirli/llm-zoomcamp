import streamlit as st
from assistant import create_assistant
from db_save import save_conversation
from db_feedback import save_feedback

assistant = create_assistant()

st.title("Course Assistant")

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    with st.spinner("Processing..."):
        answer = assistant.rag(user_input)
        st.success("Completed!")
        st.write(answer)

        record = assistant.last_call
        st.write(f"Response time: {record.response_time:.2f}s")
        st.write(f"Prompt tokens: {record.prompt_tokens}")
        st.write(f"Completion tokens: {record.completion_tokens}")
        st.write(f"Cost: ${record.cost:.4f}")

        conversation_id = save_conversation(record, user_input, "llm-zoomcamp")
        st.session_state.conversation_id = conversation_id

col1, col2 = st.columns(2)
with col1:
    if st.button("+1"):
        cid = st.session_state.conversation_id
        save_feedback(cid, "user", score=1)
        st.write("Thanks!")

with col2:
    if st.button("-1"):
        cid = st.session_state.conversation_id
        save_feedback(cid, "user", score=-1)
        st.write("Thanks for the feedback!")