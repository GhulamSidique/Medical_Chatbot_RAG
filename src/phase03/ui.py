import streamlit as st

def ui():
    st.title("Ask Medical Chatbot")

    if "message" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your query here!")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({"role":"user", "content":prompt})

        resp = "hi dear"
        st.chat_message("assistant").markdown(resp)  
        st.session_state.messages.append({"role":"assistant", "content":resp})

if __name__=="__main__":
    ui()

