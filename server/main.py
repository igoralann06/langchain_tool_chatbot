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
customers = {}
orders = {}

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

                  print(param_name)

                  # API Call
                  try:
                    url = f"http://217.154.69.224/chatapi/server.php?endpoint={tool_name}&orderID=1234"
                    # response = requests.get(url)
                  except:
                     print("")

                  if customers[param_value]:
                    if param_value is None:
                        return f"Error: Missing required parameter '{param_name}'"
                    else:
                        if(tool_name == "getOpenedOrders"):
                            if(orders[param_value]):
                                tool_answer = f"Folgende Order-IDs sind möglich: {','.join(orders[param_value])}"
                            else:
                                tool_answer = f"Keine Bestellungen"
                            print(tool_answer)
                            return tool_answer
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
  userQuestion: str

def fallback_handler(userQuestion) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "system", "content": """ 
                    Bitte verwende deutsche Quellen. Basiere deine Antworten hauptsächlich auf Informationen aus Deutschland. Bitte verwende nur Informationen aus Deutschland. 
                    Antworten basieren nur auf deutschen Quellen. 
                    Bitte setze den folgenden Prompt in Bezug zu Fenster- und Türbeschlägen.
                    Sie müssen so gut wie möglich mit dem ChatGPT-Ergebnis antworten.
                    Prozess und präsentieren Sie die Antworten auf die folgenden Fragen auf Deutsch
                    Die Antwort muss nur die HTML-Struktur aufweisen und Absatz- und Überschriftenstile enthalten (nur h4, h5, h6).
                    Sie erhalten Beobachterdaten im HTML-Format.
                    Behalten Sie die ursprüngliche HTML-Struktur in der endgültigen Antwort bei. Entfernen Sie keine HTML-Tags.
                    """}, {"role": "user", "content": userQuestion}])
    return response["choices"][0]["message"]["content"]
    # return response["choices"][0]["message"]["content"]

fallback_tool = StructuredTool(
  name="General",
  description="User's general question",
  func=fallback_handler,
  args_schema=fallbackModel
)

tools.append(fallback_tool)

# contact support
class contactSupportModel(BaseModel):
  channelID: str

def contactSupport_handler(channelID: str) -> str:
    add_question(channelID)
    return """
        Okay, ich leite Sie zu unserem Kundendienstmitarbeiter weiter.
    """

contactsupport_tool = StructuredTool(
  name="Contact Support",
  description="Contact Support",
  func=contactSupport_handler,
  args_schema=contactSupportModel
)

tools.append(contactsupport_tool)

# =====================
#   Initialize LLM & Agent
# =====================

llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

agent = initialize_agent(
  tools=tools,
  llm=llm,
  agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
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
    """
    prompt = f"""
        Übersetzen Sie den folgenden Text ins Deutsche.
        Nachfolgend der Text.
        {text}
    """
    return llm.predict(prompt)


async def chat(user_message, chat_histories, customerID, orderIDs, sid):
    global tool_answer
    histories = ""
    faq_histories = ""

    for history in chat_histories:
        histories = f"{histories} {history['role']}:{history['message']}"
        if(history['role'] == 'user'):
            faq_histories = faq_histories + history['message'] + "\n"

    print(histories)

    prompt = f"""
        Dies sind die Fragen, die der Benutzer zuvor gestellt hat:
        {faq_histories}
        Dies ist die Frage, die der Benutzer jetzt gestellt hat
        {user_message}
        Bitte antworten Sie auf die aktuelle Frage
    """
    translated_query = await asyncio.to_thread(translate_to_german, prompt)

    similar_faqs = await asyncio.to_thread(
        lambda: vector_db.similarity_search_with_score(translated_query, k=5)
    )

    SIMILARITY_THRESHOLD = 0.3

    most_similar_faq = min(similar_faqs, key=lambda x: x[1])
    if most_similar_faq[1] < SIMILARITY_THRESHOLD:
        faq_content = []
        
        for faq, score in similar_faqs:
            if(score - most_similar_faq[1] < 0.05):
                faq_content.append(f"{faq.page_content} : {faq.metadata['answer']}")

        print(faq_content)
        
        prompt = PromptTemplate.from_template(
            """
            Der Benutzer hat gefragt: {user_query}

            Hier ist die relevanteste QA aus den FAQs:

            {faq_content}

            Bitte geben Sie eine klare und informative Antwort auf Englisch und verwenden Sie dabei nur die Informationen aus den FAQ und Chatverläufen.

            Befolgen Sie diese Richtlinien:
            - Verwenden Sie Symbole oder Icons, um die Antwort leicht verständlich zu machen.
            - Priorisieren Sie Informationen, die für Deutschland relevant sind.
            - Antworten Sie nur mit HTML-Inhalten für eine bessere Lesbarkeit.
            - Wenn die Informationen unvollständig oder nicht verfügbar sind, antworten Sie mit: Ich bin nicht sicher
            """
        )

        response = await asyncio.to_thread(
            lambda: llm.predict(prompt.format(user_query=translated_query, faq_content=faq_content))
        )

        if "ich bin mir nicht sicher" in response.lower():
            return await asyncio.to_thread(stylize, "Es tut mir leid, ich kann die Antwort nicht finden. Bitte kontaktieren Sie unseren Kundensupport")
            
        final_response = await asyncio.to_thread(stylize, response)

        return final_response

    async def process_request():
        global tool_answer
        translated_histories = await asyncio.to_thread(translate_to_german, histories)
        
        response = await asyncio.to_thread(agent.run, f"""
            Du bist der virtuelle Assistent für den Ventano-Support. Wenn er Hallo sagt, müssen Sie sagen: „Hallo! Ich bin der virtuelle Assistent von Ventano.“
            Bitte verwenden Sie deutsche Quellen. 
            Basieren Sie Ihre Antworten hauptsächlich auf Informationen aus Deutschland. 
            Bitte verwenden Sie nur Informationen aus Deutschland. 
            Antworten basieren nur auf deutschen Quellen. 
            Senden Sie die Antwort nur mit der HTML-Struktur zurück. 
            Sie sprechen auf Kanal (ID:{sid})
            Sie erhalten Beobachterdaten im HTML-Format. 
            Behalten Sie die ursprüngliche HTML-Struktur in der endgültigen Antwort bei. Entfernen Sie keine HTML-Tags.
            Wenn Sie die Antwort auf die Frage des Benutzers nicht kennen, müssen Sie sagen: „Ich bin mir nicht sicher“
            oder Sie können Fragen stellen, um den Support zu kontaktieren

            Dies ist der Chatverlauf
            {translated_histories}

            Benutzer fragte:
            {translated_query}
            """)

        if tool_answer:
            response = tool_answer
            tool_answer = ""

        print("result: " + response.lower())
        # Handoff logic: If the AI is not confident, escalate to human agent
        if "ich bin mir nicht sicher" in response.lower():
            print("Ich!!!")
            return await asyncio.to_thread(translate_to_german, "Es tut mir leid, ich kann die Antwort nicht finden. Bitte kontaktieren Sie unseren Kundensupport")

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
        chat_histories = data['chatHistories']
        customerID = data['customerID']
        orderIDs = data['orderIDs']
        customers[room] = customerID
        orders[room] = orderIDs
        print(f"Message from {sid} in room {room}: {message}")
        add_chat(message, sid, False)
        await sio.emit('message', {'sid': sid, 'message': message, 'bot': get_bot_by_id(sid)}, room=room)

        bot = get_bot_by_id(room)
        if not bot:
            answer = await chat(message, chat_histories, customerID, orderIDs, sid)
            print(answer)
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

