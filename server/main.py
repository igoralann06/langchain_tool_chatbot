from fastapi import FastAPI, Request
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
import openai

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"], 
  allow_credentials=True,
  allow_methods=["*"],  
  allow_headers=["*"], 
)

client_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../client"))
app.mount("/board", StaticFiles(directory=client_path, html=True), name="client")

# =====================
#  Load and Index FAQ
# =====================
df = pd.read_csv("faq.csv", encoding="windows-1252", header=None, names=["question", "answer"])
embedding_model = OpenAIEmbeddings()

documents = [
    Document(page_content=row["question"], metadata={"answer": row["answer"]})
    for _, row in df.iterrows()
]

vector_db = FAISS.from_documents(documents, embedding_model)

# =====================
#   Define Tools
# =====================

class fallbackModel(BaseModel):
  query: str

class TrackParcel(BaseModel):
  productName: str

class ConfirmPaymentInput(BaseModel):
  order_id: str

class CheckReturnStatusInput(BaseModel):
  return_id: str

class GetShippingCostInput(BaseModel):
  destination: str

def fallback_handler(query: str) -> str:
   response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}])
   return response["choices"][0]["message"]["content"]

def trackParcel(order_id: str) -> str:
  return "Say that 'trackParcel' API is called"

def confirm_payment(order_id: str) -> str:
  return "Say that 'confirm_payment' API is called"

def check_return_status(order_id: str) -> str:
  return "Say that 'check_return_status' API is called."

def get_shipping_cost(order_id: str) -> str:
  return "Say that 'get_shipping_cost' API is called"

fallback_tool = StructuredTool(
  name="General question",
  description="Handles ANY query NOT related to parcel tracking, payment confirmation, return status, or shipping costs.",
  func=fallback_handler,
  args_schema=fallbackModel
)

track_tool = StructuredTool(
  name="TrackParcel",
  description="Track product",
  func=trackParcel,
  args_schema=TrackParcel
)

payment_tool = StructuredTool(
  name="ConfirmPayment",
  description="Confirm if payment has been received ONLY with an order ID.",
  func=confirm_payment,
  args_schema=ConfirmPaymentInput
)

return_tool = StructuredTool(
  name="CheckReturnStatus",
  description="Check the status of a return using order ID.",
  func=check_return_status,
  args_schema=CheckReturnStatusInput
)

shipping_tool = StructuredTool(
  name="GetShippingCost",
  description="Get shipping cost for a order ID.",
  func=get_shipping_cost,
  args_schema=GetShippingCostInput
)

# =====================
#   Initialize LLM & Agent
# =====================

llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

agent = initialize_agent(
  tools=[track_tool, payment_tool, return_tool, shipping_tool, fallback_tool],
  llm=llm,
  agent_type=AgentType.OPENAI_MULTI_FUNCTIONS,
  verbose=True,
  handle_parsing_errors=True,
  max_iterations=3
)

def translate_to_english(text: str) -> str:
  """Translate German text to English using GPT-4."""
  prompt = f"Translate the following German text to English: {text}"
  return llm.predict(prompt)

def translate_to_german(text: str) -> str:
  """Translate English text to German using GPT-4."""
  prompt = f"Translate the following English text to German: {text}"
  return llm.predict(prompt)

# =====================
#   FastAPI Chat Route
# =====================

@app.post("/chat")
async def chat(request: Request):
  data = await request.json()
  user_message = data.get("message")

  # Translate German input to English
  translated_query = translate_to_english(user_message)

  # Search FAQ database with the translated query
  similar_faqs = vector_db.similarity_search_with_score(translated_query, k=1)

  SIMILARITY_THRESHOLD = 0.35
  print(similar_faqs[0][1])

  if similar_faqs and similar_faqs[0][1] < SIMILARITY_THRESHOLD:  # Lower scores mean better similarity
    best_match, score = similar_faqs[0]
    faq_answer = best_match.metadata["answer"]

    # Translate FAQ answer back to German
    prompt = PromptTemplate.from_template(
      "Here is a frequently asked question: {faq}\nAnswer: {answer}\nRephrase the answer in a friendly tone. Please answer in German."
    )
    response = llm.predict(prompt.format(faq=best_match.page_content, answer=faq_answer))

    return {"response": response}

  # If no FAQ match is found, use structured tools or LLM
  response = agent.run(translated_query)

  # Translate final response back to German before returning
  final_response = translate_to_german(response)

  return {"response": final_response}
