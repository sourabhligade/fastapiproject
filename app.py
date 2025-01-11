import streamlit as st
import requests
import base64

st.set_page_config(layout="wide", page_title="PDF Chat")

st.markdown("""
<style>
    .pdf-container {
        height: calc(100vh - 100px);
        border: 1px solid #ddd;
        border-radius: 10px;
        overflow: hidden;
    }
    .pdf-container iframe {
        width: 100%;
        height: 100%;
        border: none;
    }
    .qa-pair {
        margin: 10px 0;
        padding: 10px;
    }
    .question {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .answer {
        margin-left: 20px;
    }
</style>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

def display_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    return f'<div class="pdf-container"><iframe src="data:application/pdf;base64,{base64_pdf}"></iframe></div>'

col1, col2 = st.columns([0.6, 0.4])

with col1:
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    if uploaded_file:
        st.markdown(display_pdf(uploaded_file), unsafe_allow_html=True)
        if 'asset_id' not in st.session_state:
            with st.spinner("Processing document..."):
                uploaded_file.seek(0)
                response = requests.post('http://localhost:8000/upload/', files={'file': uploaded_file})
                if response.ok:
                    st.session_state.asset_id = response.json()['asset_id']

with col2:
    if 'asset_id' in st.session_state:
        question = st.text_input("", placeholder="Ask me a question about the PDF...", key="question_input")
        if st.button("Send"):
            if question:
                # Clear previous messages after each new Q&A interaction
                st.session_state.messages.clear()

                st.session_state.messages.append({"role": "user", "content": question})
                with st.spinner("Getting answer..."):
                    response = requests.post(
                        f"http://localhost:8000/ask/{st.session_state.asset_id}",
                        json={"question": question}
                    )
                    if response.ok:
                        data = response.json()
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data['answer'],
                            "chunks": data['relevant_chunks']
                        })
                        st.rerun()  # This will clear the session state and rerun the app

    # Display the latest question and answer
    if len(st.session_state.messages) > 0:
        st.markdown(f"""
            <div class="qa-pair">
                <div class="question">Q: {st.session_state.messages[-2]['content']}</div>
                <div class="answer">A: {st.session_state.messages[-1]['content']}</div>
            </div>
        """, unsafe_allow_html=True)
