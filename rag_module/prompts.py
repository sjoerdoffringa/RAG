prompts = {
    "vanilla-rag":
    """
    You are an AI assistant. Use the following retrieved context to answer the question.

    Context:
    {joined_chunks}

    Question:
    {query}
    """,

    "self-citation":
    """
    Write a high-quality answer for the given question using the provided documents and cite them properly using [1][2][3].

    Documents:
    {joined_chunks}

    Question:
    {query}
    """,

    "counterfactual-judge":
    """
    The provided documents cannot properly answer the question.
    If the topic of the question is not at all related to the documents, reply with "No related documents found".
    If the topic of the question is somewhat related to the documents, provide a similar question that can be answered using the documents.
    Reply only with the "No related documents found" or a question and nothing else.

    Documents:
    {joined_chunks}

    Question:
    {query}
    """,

    "counterfactual":
    """
    The provided documents cannot properly answer the question.
    Evaluate the documents and provide a similar question that can be answered using the documents.
    Reply only with a question and nothing else.
    
    Documents:
    {joined_chunks}

    Question:
    {query}
    """
}