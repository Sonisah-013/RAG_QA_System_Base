"""
prompts.py
Prompt template used to ground the LLM's answer in retrieved context.
"""

from langchain_core.prompts import PromptTemplate

QA_PROMPT_TEMPLATE = """Answer the question using ONLY the context below.

Rules:
- For "what is" or "define" questions, write 5 to 6 short lines: line 1 is the definition in one clear sentence, and lines 2 to 6 are the most important points about it, one per line.
- For questions about features, Types, advantages, disadvantages, benefits, characteristics, components, applications, uses, examples, types, steps or differences:
  - Start with one short sentence that says what they are.
  - Then give EVERY item from the context as a separate bullet point, in this form:
    - **Name:** one simple sentence explaining it.
- Stay on the topic of the question. Do not add anything unrelated to it.
- Use simple words. Do not copy long text word for word.
- Never mention page numbers, unit numbers, headings, author names, "the context", "the document" or "the PDF".
- If the context has fewer relevant points than expected, give only those and do not pad the answer.
- If the answer is not in the context at all, reply exactly: "I don't have enough information in the document to answer that."
- Never make up information.

Context:
{context}

Question: {input}

Answer:"""
qa_prompt = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "input"],
)


if __name__ == "__main__":
    example = qa_prompt.format(
        context="The Eiffel Tower was completed in 1889.",
        input="When was the Eiffel Tower completed?",
    )
    print(example)