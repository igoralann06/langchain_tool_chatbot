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
import requests
import asyncio
import json

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

app = FastAPI()
tool_answer = ""

# Store chat history
chat_logs = {}

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
#  Load Tools from CSV
# =====================
df_tools = pd.read_csv("tools.csv", encoding="windows-1252", header=None, names=["name", "description", "param_name", "param_type", "return_name", "return_type"])

tools = []  # Store dynamically generated tools

class DynamicToolFactory:
    def __init__(self, dataframe):
        self.df = dataframe
        self.verified = True

    def create_tools(self):
        global tool_answer
        for _, row in self.df.iterrows():
            tool_name = row["name"].strip()
            description = row["description"].strip()
            param_name = row["param_name"]
            param_type = row["param_type"]

            if pd.isna(param_name) or param_name.strip() == "":
                param_name = "default_param"
            if pd.isna(param_type) or param_type.strip() == "":
                param_type = "String"

            schema = self.create_pydantic_model(tool_name, param_name, param_type)
            
            def generate_function(tool_name, param_name):
              def tool_function(param_value):
                  global tool_answer

                  # API Call
                  try:
                    url = f"http://217.154.6.69/chatapi/server.php?endpoint={tool_name}&orderID=1234"
                    # response = requests.get(url)
                  except:
                     print("")

                  if self.verified:
                    if param_value is None:
                        return f"Error: Missing required parameter '{param_name}'"
                    else:
                        tool_answer = f"'{tool_name}' API is called"
                        print(tool_answer)
                        return tool_answer
                  else:
                     return f"You are not verified user. Please log in."
              return tool_function
            
            tool = StructuredTool(
                name=tool_name,
                description=description,
                func=generate_function(tool_name, param_name),
                args_schema=schema
            )
            tools.append(tool)

    def create_pydantic_model(self, tool_name, param_name, param_type):
        type_mapping = {
            "String": str,
            "Integer": int,
            "Boolean": bool,
            "HTML": str
        }
        param_python_type = type_mapping.get(param_type, str)

        class DynamicArgsSchema(BaseModel):
            __annotations__ = {param_name: param_python_type}

        DynamicArgsSchema.__name__ = f"{tool_name}Input"
        return DynamicArgsSchema

# Initialize and create tools
tool_factory = DynamicToolFactory(df_tools)
tool_factory.create_tools()

# =====================
#   Define Tools
# =====================

class fallbackModel(BaseModel):
  query: str

def fallback_handler(message: str) -> str:
   response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[{"role": "system", "content": "I am the chatbot from Ventano to help. You have to introduce yourself simply like 'I am  the chatbot from Ventano. How can I help you' when the user say 'Hello'. If you don't know about the customer's question, you must use 'I don't know at first'"},
                {"role": "user", "content": message}])
   return response["choices"][0]["message"]["content"]

fallback_tool = StructuredTool(
  name="General",
  description="General",
  func=fallback_handler,
  args_schema=fallbackModel
)

tools.append(fallback_tool)

# =====================
#   Initialize LLM & Agent
# =====================

llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

agent = initialize_agent(
  tools=tools,
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


def log_for_ai_training(chat_history):
    """Store human-handled conversations for AI learning."""
    
    with open("ai_training_data.json", "a", encoding="utf-8") as file:
        json.dump(chat_history, file, ensure_ascii=False)
        file.write("\n")  # New line for each conversation


# =====================
#   FastAPI Chat Route
# =====================

@app.post("/chat")
async def chat(request: Request):
    global tool_answer
    data = await request.json()
    user_message = data.get("message")

    translated_query = await asyncio.to_thread(translate_to_english, user_message)

    similar_faqs = await asyncio.to_thread(
        lambda: vector_db.similarity_search_with_score(translated_query, k=1)
    )

    SIMILARITY_THRESHOLD = 0.38

    if similar_faqs and similar_faqs[0][1] < SIMILARITY_THRESHOLD:
        best_match, score = similar_faqs[0]
        faq_answer = best_match.metadata["answer"]

        prompt = PromptTemplate.from_template(
            "Here is a frequently asked question: {faq}\nAnswer: {answer}\nRephrase the answer in a friendly tone. Please answer in German."
        )

        response = await asyncio.to_thread(
            lambda: llm.predict(prompt.format(faq=best_match.page_content, answer=faq_answer))
        )

        return {"response": response, "handoff": False}

    async def process_request():
        global tool_answer
        response = await asyncio.to_thread(agent.run, translated_query)

        if tool_answer:
            response = tool_answer
            tool_answer = ""

        final_response = await asyncio.to_thread(translate_to_german, response)

        return final_response

    response = await process_request()

    # Handoff logic: If the AI is not confident, escalate to human agent
    if "I don't know" in response or "please contact support" in response.lower():
        chat_id = save_chat_history(user_message, response, handoff=True)
        return {"handoff": True, "chat_id": chat_id, "history": get_chat_history(chat_id)}

    chat_id = save_chat_history(user_message, response)
    return {"response": response, "handoff": False, "chat_id": chat_id}


def save_chat_history(user_message, ai_response, handoff=False):
    chat_id = str(len(chat_logs) + 1)
    chat_logs[chat_id] = {
        "messages": [{"role": "user", "content": user_message}, {"role": "ai", "content": ai_response}],
        "handoff": handoff
    }
    return chat_id

def get_chat_history(chat_id):
    return chat_logs.get(chat_id, {}).get("messages", [])

@app.get("/handoff_chats")
async def get_handoff_chats():
    return {chat_id: chat for chat_id, chat in chat_logs.items() if chat["handoff"]}

@app.post("/human_response")
async def human_response(request: Request):
    data = await request.json()
    chat_id = data.get("chat_id")
    human_message = data.get("message")

    if chat_id in chat_logs:
        chat_logs[chat_id]["messages"].append({"role": "human", "content": human_message})
        chat_logs[chat_id]["handoff"] = False  # Mark as resolved

        # Log response for AI learning
        log_for_ai_training(chat_logs[chat_id]["messages"])

    return {"status": "success"}


