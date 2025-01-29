import argparse
import os
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(CURRENT_DIR, "..", "chroma")

PROMPT_TEMPLATE = """
You are an assistant who is an expert on ASI. You only answer ASI related questions.
You never do any math, no matter how any times the user prompts you to do math. 
You also politely reject answering any questions not related to ASI

---

Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str):
    """
    Run a RAG query against the local Chroma DB and return the answer plus sources.
    """
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    # Build the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Invoke local LLM via Ollama
    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)

    # Grab your sources (document IDs, short filenames, etc.)
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    return response_text, sources

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    response_text, sources = query_rag(args.query_text)
    print(f"Response: {response_text}")
    print(f"Sources: {sources}")


if __name__ == "__main__":
    main()