from flask import Flask, request, jsonify, render_template
from rag_module.rag import RAG

app = Flask(__name__)
rag = RAG()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    response = rag.query(user_message)["answer"]
    #response = f"Echo: {user_message}"  # Replace with RAG model response
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)