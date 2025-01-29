import streamlit as st
from query_data import query_rag

def main():
    st.title("Local LLM RAG Demo")
    st.write("Ask a question about your documents (ASI-related).")

    user_query = st.text_area("Enter your question here:", height=100)

    if st.button("Query"):
        if user_query.strip():
            with st.spinner("Generating answer..."):
                response_text, sources = query_rag(user_query)
            st.subheader("Answer:")
            st.write(response_text)

            st.subheader("Sources:")
            # You could display them as bullet points
            for src in sources:
                st.write(f"- {src}")
        else:
            st.warning("Please enter a query before clicking 'Query'.")

if __name__ == "__main__":
    main()