📚 **Intelligent FAQ Chatbot**
This project is a Streamlit-based FAQ chatbot that allows users to upload a .pdf or .txt file and interact with it through natural language questions. The chatbot extracts the document content, indexes it using semantic embeddings, and provides context-aware answers to user queries.

🔍 **Overview**
Uses Sentence Transformers (all-MiniLM-L6-v2) to convert text chunks into vector embeddings.

Employs FAISS for fast similarity search over the document content.

Answers are generated using a lightweight transformers-based QA model (distilbert-base-cased-distilled-squad).

UI is built with Streamlit, featuring a simple document uploader, chat interface, and sidebar for history management.

⚙️ **Setup Instructions**
1. Clone the repository:
   git clone https://github.com/yourusername/intelligent-faq-chatbot.git
   cd intelligent-faq-chatbot
2. Set up the environment:
   conda create -n faqbot python=3.10
   conda activate faqbot
3. Install dependencies:
   pip install -r requirements.txt
4. Run the application:
   streamlit run faq_chat_ui.py

💡 **Features**
Upload .pdf or .txt files and process their content.

Ask any question based on the uploaded content.

Vector-based search ensures semantically relevant context is retrieved.

Sidebar includes:

Chat history (collapsible for longer chats)

+ button to upload new documents

🗑️ "Clear Chat" button to reset the session

✅ Assumptions
The input documents are in English and contain well-structured text.

For simplicity, FAISS is configured using a flat L2 index (good for small to medium-sized documents).

Embedding and QA models are optimized for speed and do not require a GPU.

Chat history is stored locally for the current session (extendable to persistent storage or database if needed).
