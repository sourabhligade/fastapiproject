import streamlit as st
import requests
import base64

# Set the page layout and title
st.set_page_config(layout="wide", page_title="Chat with Any PDF")

# Custom styling with improved fonts and responsive design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Body Styling */
    body {
        background-color: #ffffff;
        color: #1a1a1a;
        line-height: 1.6;
    }

    /* Header Styling */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .logo {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #5F4B8B, #4A3F73);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .title {
        font-size: 3em;
        font-weight: 700;
        background: linear-gradient(135deg, #5F4B8B, #4A3F73);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }

    .subtitle {
        font-size: 1.1em;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }

    /* PDF Container */
    .pdf-container {
        height: 600px;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
    }
    
    .pdf-container iframe {
        width: 100%;
        height: 100%;
        border: none;
    }

    /* Q&A Section */
    .qa-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .qa-pair {
        margin: 1rem 0;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #5F4B8B;
    }
    
    .question {
        font-weight: 600;
        color: #5F4B8B;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 1em; /* Reduced font size for the question */
    }
    
    .answer {
        color: #333;
        font-size: 0.9rem; /* Reduced font size for the answer */
        line-height: 1.6;
        padding-left: 2rem;
    }

    /* Loading Spinner */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0,0,0,.1);
        border-radius: 50%;
        border-top-color: #5F4B8B;
        animation: spin 1s ease-in-out infinite;
    }

    .loading-text {
        font-size: 1.2em;
        color: #5F4B8B;
        margin-top: 1rem;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Footer Styling */
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #666;
        font-size: 1rem;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .title { font-size: 2em; }
        .subtitle { font-size: 1em; }
        .step-card { flex-direction: column; }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-container"><div class="logo">📚</div><div class="title">Chat with Any PDF</div></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Transform your PDF reading experience with AI-powered conversations. Get instant answers, summaries, and insights from your documents.</div>', unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

def display_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    return f'<div class="pdf-container"><iframe src="data:application/pdf;base64,{base64_pdf}"></iframe></div>'

# Main content
col1, col2 = st.columns([0.6, 0.4])

with col1:
    uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
    if uploaded_file:
        st.markdown(display_pdf(uploaded_file), unsafe_allow_html=True)
        if 'asset_id' not in st.session_state:
            with st.spinner("Processing document..."):
                uploaded_file.seek(0)
                response = requests.post('http://localhost:8000/upload/', files={'file': uploaded_file})
                if response.ok:
                    st.session_state.asset_id = response.json()['asset_id']
                    # Simulate predefined questions generation (you can make this dynamic by analyzing the PDF)
                    predefined_questions = [
                        "What is the summary of the document?",
                        "What are the key points in the document?",
                        "Can you explain the main argument of the document?",
                        "What is the conclusion of the document?"
                    ]
                    st.session_state.predefined_questions = predefined_questions

with col2:
    if 'asset_id' in st.session_state:
        question = st.text_input("", key="question_input", placeholder="Ask a question about this document...")
        
        if st.button("Ask", key="send_button", type="primary"):
            if question:
                st.session_state.messages.append({"role": "user", "content": question})
                with st.spinner("Thinking..."):
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
                        st.rerun()

        # Display questions without the "Predefined Questions" header, make text purple and sleek
        if 'predefined_questions' in st.session_state:
            for q in st.session_state.predefined_questions:
                if st.button(q, key=q):
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner("Thinking..."):
                        response = requests.post(
                            f"http://localhost:8000/ask/{st.session_state.asset_id}",
                            json={"question": q}
                        )
                        if response.ok:
                            data = response.json()
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": data['answer'],
                                "chunks": data['relevant_chunks']
                            })
                            st.rerun()

    # Display chat history
    if st.session_state.messages:
        st.markdown("### Conversation")
        for msg in st.session_state.messages[-2:]:  # Show last Q&A pair
            if msg["role"] == "user":
                st.markdown(f"""
                    <div class="qa-pair">
                        <div class="question">
                            👤 {msg['content']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="qa-pair">
                        <div class="answer">
                            🤖 {msg['content']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

if uploaded_file: 
    # Footer
    st.markdown("""
    <div class="footer">
        Powered with AI 💡 | Built using Streamlit 💻 | Chat with any PDF 📝
    </div>
    """, unsafe_allow_html=True)
