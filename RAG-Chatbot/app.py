import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
DB_PATH = ".chroma_db"

def initialize_rag_backend():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, None, "Missing GEMINI_API_KEY environment credentials."
    
    os.environ["GOOGLE_API_KEY"] = api_key
    
    if not os.path.exists(DB_PATH):
        return None, None, "Persistent vector directory not found. Please run ingest.py first."
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    return retriever, llm, None

retriever, llm, init_error = initialize_rag_backend()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    if init_error:
        return jsonify({"error": init_error}), 500
        
    data = request.get_json()
    user_query = data.get("query")
    
    if not user_query:
        return jsonify({"error": "No query provided."}), 400
        
    try:
        matched_docs = retriever.invoke(user_query)
        context_text = "\n\n".join([doc.page_content for doc in matched_docs])
        
        prompt_template = (
            f"You are a precise document analysis assistant.\n"
            f"Answer the question based strictly on the provided context. If the answer isn't present, "
            f"say you don't know.\n\n"
            f"--- CONTEXT ---\n{context_text}\n---------------\n\n"
            f"Question: {user_query}\n"
            f"Answer:"
        )
        
        execution_result = llm.invoke(prompt_template)
        output_text = execution_result.content
        
        return jsonify({"response": output_text})
        
    except Exception as e:
        return jsonify({"error": f"An execution error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))