import pandas as pd
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Read CSV without headers and assign column names manually
df = pd.read_csv("faq.csv", encoding="windows-1252", header=None, names=["question", "answer"])

# Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings()

# Create FAISS index with questions and store answers in metadata
documents = [
    Document(page_content=row["question"], metadata={"answer": row["answer"]})
    for _, row in df.iterrows()
]

vector_db = FAISS.from_documents(documents, embedding_model)

# Example User Query
user_query = "Machen Sie auch Nachfertigungen?"
similar_faqs = vector_db.similarity_search(user_query)

# Retrieve the most relevant answer
if similar_faqs:
    best_match = similar_faqs[0]
    print("Matched FAQ:", best_match.page_content)
    print("Answer:", best_match.metadata["answer"])
else:
    print("No matching FAQ found.")

# Use LLM to refine the answer
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)
prompt = PromptTemplate.from_template(
    "Here is a frequently asked question: {faq}\nAnswer: {answer}\nRephrase the answer in a friendly tone."
)
response = llm.predict(prompt.format(faq=best_match.page_content, answer=best_match.metadata["answer"]))

print("Rephrased Answer:", response)
