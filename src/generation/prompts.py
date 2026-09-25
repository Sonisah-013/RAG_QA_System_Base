"""
prompts.py
Prompt template used to ground the LLM's answer in retrieved context.
"""

from langchain_core.prompts import PromptTemplate

QA_PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below.
If the answer is not contained in the context, say "I don't have enough information in the document to answer that."
Do not make up information that isn't supported by the context. Keep answers concise and direct.

Context:
{context}

Question: {question}

Answer:"""

qa_prompt = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


if __name__ == "__main__":
    # Quick sanity check that the template renders correctly
    example = qa_prompt.format(
        context="The Eiffel Tower was completed in 1889.",
        question="When was the Eiffel Tower completed?",
    )
    print(example)