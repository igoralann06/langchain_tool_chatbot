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
#  Load Tools from CSV
# =====================
df_tools = pd.read_csv("tools.csv", encoding="windows-1252", header=None, names=["name", "description", "param_name", "param_type", "return_name", "return_type"])

tools = []  # Store dynamically generated tools

class DynamicToolFactory:
    def __init__(self, dataframe):
        self.df = dataframe
        self.verified = True

    def create_tools(self):
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
                  if self.verified:
                    if param_value is None:
                        return f"Error: Missing required parameter '{param_name}'"
                    else:
                        return f"Say that '{tool_name}' API is called"
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

def fallback_handler(query: str) -> str:
   response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}])
   return response["choices"][0]["message"]["content"]

fallback_tool = StructuredTool(
  name="General question",
  description="General question",
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

# =====================
#   FastAPI Chat Route
# =====================

@app.post("/chat")
async def chat(request: Request):
  data = await request.json()
  user_message = data.get("message")

  # Translate German input to English
  translated_query = "The orderId is s-2039." + translate_to_english(user_message)

  # Search FAQ database with the translated query
  similar_faqs = vector_db.similarity_search_with_score(translated_query, k=1)

  SIMILARITY_THRESHOLD = 0.38
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
