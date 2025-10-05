import streamlit as st
import requests

st.set_page_config(page_title="Patent Assistant", page_icon="📄")

# API configuration
API_URL = "http://api:8000/query"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about patents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Call your RAG API
            response = requests.post(
                API_URL,
                json={"query": prompt, "use_rag": True},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                full_response = response.json().get("response", "No response from API")
            else:
                full_response = f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            full_response = f"Error connecting to API: {str(e)}"
        
        # Display assistant response
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})