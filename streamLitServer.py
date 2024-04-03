import streamlit as st
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from ProcessInput.UserInput import processUserInput
import json


st.set_page_config(page_title="ChatBot",page_icon="🤖")

st.title("DineBot")

if "messages" not in st.session_state:
    st.session_state.messages = []


#conversation
for message in st.session_state.messages:
    if isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)


#user input
user_query = st.chat_input("Your message")
if user_query:
    with st.chat_message("Human"):
        st.markdown(user_query)
    st.session_state.messages.append(HumanMessage(user_query))

    with st.chat_message("AI"):
        response = processUserInput(user_query)
        responseObject = json.dumps(response)
        print(responseObject)
        st.markdown(response)
    st.session_state.messages.append(AIMessage(response))

