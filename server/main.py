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
import asyncio
import json
from fastapi.responses import StreamingResponse
import csv
from io import StringIO

from db_questions import add_question, get_questions, init_db, delete_question
from db_chats import add_chat, get_chats, init_chats, get_question
from db_bots import add_bot, delete_bot, get_bot_by_id, init_bot
from db_faqs import init_faq, add_faq, get_faqs

# socket.io
import socketio

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

app = FastAPI()
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
socket_app = socketio.ASGIApp(sio, app)

tool_answer = ""

# Store chat history
chat_logs = {}

init_db()
init_chats()
init_bot()
init_faq()

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

faqs = get_faqs()

for faq in faqs:
    new_doc = Document(page_content=faq["question"], metadata={"answer": faq["answer"]})
    vector_db.add_documents([new_doc])


# =====================
#  Load Tools from CSV
# =====================
df_tools = pd.read_csv("tools.csv", encoding="windows-1252", header=None, names=["name", "description", "param_name", "param_type", "return_name", "return_type"])

tools = []  # Store dynamically generated tools
adminSid = ""

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
      model="gpt-4o-mini",
      messages=[{"role": "system", "content": "Ich bin der Chatbot von Ventano und bin hier, um Ihnen zu helfen. Wenn der Benutzer Sie begrüßt, antworten Sie freundlich und stellen Sie sich vor, indem Sie etwas sagen wie: „Hallo, ich bin der Chatbot von Ventano. Wie kann ich Ihnen heute helfen? Bitte nutze deutsche Quellen. Beziehe deine Antworten vorrangig auf Informationen aus Deutschland. Bitte nur Informationen aus Deutschland verwenden. Antworten nur auf Basis von deutschen Quellen."},
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
  """Übersetzen Sie englischen Text mit GPT-4 ins Deutsche."""
  prompt = f"Übersetzen Sie den folgenden englischen Text ins Deutsche: {text}"
  return llm.predict(prompt)

def stylize(text: str) -> str:
    """
        Bitte stilisieren Sie den obigen Text mit Symbolen, damit er leicht lesbar ist.
    """
    prompt = f"""
        Übersetzen Sie den folgenden Text ins Deutsche.
        Nachfolgend der Text.
        {text}
    """
    return llm.predict(prompt)


async def chat(user_message, sid):
    global tool_answer

    translated_query = await asyncio.to_thread(translate_to_german, user_message)

    similar_faqs = await asyncio.to_thread(
        lambda: vector_db.similarity_search_with_score(translated_query, k=5)
    )

    SIMILARITY_THRESHOLD = 0.38

    if similar_faqs:
        relevant_faqs = [faq for faq, score in similar_faqs if score < SIMILARITY_THRESHOLD]
        print(relevant_faqs)
        if relevant_faqs:
            faq_content = "\n".join(
                [f"{faq.page_content} : {faq.metadata['answer']}" for faq in relevant_faqs]
            )

            # Ask GPT-4 to summarize and tailor the response
            prompt = PromptTemplate.from_template(
                """
                Der Benutzer hat gefragt: {user_query}
                Hier sind die nützlichen QAs aus den FAQs:

                {faq_content}  

                Bitte geben Sie eine klare und ausreichend informative Antwort auf Englisch und verwenden Sie dabei nur die FAQ-Antwort.
                Bitte nutze deutsche Quellen.
                Beziehe deine Antworten vorrangig auf Informationen aus Deutschland.
                Bitte stilisieren Sie den obigen Text mit Symbolen, damit er leicht zu verstehen ist.
                Bitte nur Informationen aus Deutschland verwenden.
                Antworten nur auf Basis von deutschen Quellen.
                Antworten Sie nur als HTML. Listen Sie die FAQs nicht auf.
                Wenn Sie keine Informationen zu den obigen QAs vollständig kennen, müssen Sie wie folgt antworten: „Kontaktieren Sie den Ventano-Support“.
                """
            )

            response = await asyncio.to_thread(
                lambda: llm.predict(prompt.format(user_query=translated_query, faq_content=faq_content))
            )

            if "Kontaktieren Sie den Ventano-Support" in response.lower():
                add_question(sid)
                
            final_response = await asyncio.to_thread(translate_to_german, response)

            return final_response

    async def process_request():
        global tool_answer
        response = await asyncio.to_thread(agent.run, "Ich bin der Chatbot von Ventano und bin hier, um Ihnen zu helfen. Wenn der Benutzer Sie begrüßt, antworten Sie freundlich und stellen Sie sich vor, indem Sie etwas sagen wie: „Hallo, ich bin der Chatbot von Ventano. Wie kann ich Ihnen heute helfen? Bitte nutze deutsche Quellen. Beziehe deine Antworten vorrangig auf Informationen aus Deutschland. Bitte nur Informationen aus Deutschland verwenden. Antworten nur auf Basis von deutschen Quellen." + translated_query)

        if tool_answer:
            response = tool_answer
            tool_answer = ""

        # Handoff logic: If the AI is not confident, escalate to human agent
        if "Kontaktieren Sie den Ventano-Support" in response.lower():
            add_question(sid)

        final_response = await asyncio.to_thread(translate_to_german, response)

        return final_response

    response = await process_request()
    
    # print(response)
    return response

def get_chat_history(chat_id):
    return chat_logs.get(chat_id, {}).get("messages", [])

@app.get("/questions")
def get_questions_api():
    questions = get_questions()
    return questions

@app.post("/chats")
async def get_chats_api(request: Request):
    body = await request.json()
    sid = body.get("sid")
    chats = get_chats(sid)
    return chats

@app.post("/bot")
async def get_bot_api(request: Request):
    body = await request.json()
    sid = body.get("sid")
    bot = get_bot_by_id(sid)
    return {"checked": not bot}

@app.get("/download-faq/")
def download_faq():
    faq_list = get_faqs()
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["id", "question", "answer"])
    writer.writeheader()
    writer.writerows(faq_list)

    # Move the cursor to the start of the stream
    output.seek(0)

    # Return as a downloadable file
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=faq_list.csv"}
    )

@app.post("/botstatus")
async def get_bot_status(request: Request):
    data = await request.json()
    print("botstatus "+data["sid"])
    if(data["checked"]):
        delete_bot(data["sid"])
    else:
        add_bot(data["sid"])


@sio.event
async def connect(sid, environ):
    print(f"Client connected with sid: {sid}")
    # Send the sid to the client
    await sio.emit('set_sid', {'sid': sid}, to=sid)

@sio.event
async def join(sid, data):
    global adminSid
    room = data['room']
    if(room == "admin"):
        adminSid = sid
    else:
        await sio.enter_room(sid, room)
    print(f"Client {sid} joined room {room}")

@sio.event
async def message(sid, data):
    global adminSid, vector_db
    print(sid)
    if(sid != adminSid):
        room = data['room']
        message = data['message']
        print(f"Message from {sid} in room {room}: {message}")
        add_chat(message, sid, False);
        await sio.emit('message', {'sid': sid, 'message': message, 'bot': get_bot_by_id(sid)}, room=room)

        bot = get_bot_by_id(room)
        if not bot:
            answer = await chat(message, sid)
            add_chat(answer, sid, True)
            await sio.emit('message', {'sid': 'bot', 'message': answer}, room=room)
    else:
        room = data['room']
        message = data['message']

        add_chat(message, room, True);
        print(f"Message from {sid} in room {room}: {message}")
        await sio.emit('message', {'sid': sid, 'message': message}, room=room)

        questions = get_question(room)
        add_faq(questions[0]["text"], message)

        new_doc = Document(page_content=questions[0]["text"], metadata={"answer": message})
        vector_db.add_documents([new_doc])

@sio.event
async def visit(sid, data):
    print("visit "+data["room"])
    await sio.enter_room(sid, data["room"])

@sio.event
async def ignore(sid, data):
    print("ignore "+data["room"])
    delete_question(data["room"])
    delete_bot(data["room"])

@sio.event
async def bot(sid, data):
    print("bot "+data["room"])
    add_bot(data["room"])

@sio.event
async def typing(sid, data):
    print("typing "+data["sid"])
    await sio.emit("typingstatus", {"sid": data["sid"], "typing": data["typing"]}, skip_sid=sid)


@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)

