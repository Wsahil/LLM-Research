import streamlit as st

def main():
    st.title("Jarvis Chat Bot")
    # Initialize the chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Displaying all messages in the chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("What is your message?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generating and displaying chatbot response
        with st.chat_message("assistant"):
            response = f"Thank you for telling me: {prompt}"
            st.markdown(response)

        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})



if __name__ == "__main__":
    main()