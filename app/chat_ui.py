import streamlit as st
import requests

st.set_page_config(page_title="Patent Assistant", page_icon="📄")

# API configuration
API_URL = "http://ip-assistant-api:8000/query"
API_URL_STREAM = "http://ip-assistant-api:8000/query/stream"

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
            # Get model from environment variables with fallback
            import os
            model = os.getenv("MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

            # Stream from the API
            with requests.post(
                API_URL_STREAM,
                json={"query": prompt, "use_rag": True, "model": model},
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=300,
            ) as resp:
                if resp.status_code != 200:
                    full_response = f"Error: {resp.status_code} - {resp.text}"
                else:
                    for chunk in resp.iter_content(chunk_size=512):
                        if not chunk:
                            continue
                        text = chunk.decode("utf-8", errors="ignore")
                        full_response += text
                        message_placeholder.markdown(full_response)
        
        except Exception as e:
            full_response = f"Error connecting to API: {str(e)}"
        
        # Display assistant response
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
