
import uvicorn
import os
import re

from typing import Literal
from typing_extensions import TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.graph.message import add_messages


database = [
    {"code": "1123", "name": "farinha de trigo", "price": "12,00$"},
    {"code": "1124", "name": "maçã verde", "price": "4,00$kg"},
    {"code": "1125", "name": "maçã argentina", "price": "2,00$"},
    {"code": "1126", "name": "arroz branco", "price": "8,50$kg"},
]

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

crypto_key = os.getenv("ENCRYPTION_KEY")
cipher_suite = Fernet(crypto_key)

app = FastAPI(title="Store Systems LangGraph")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=api_key
)

support_texts = [
    "A política de reembolso da nossa loja permite devoluções em até 30 dias corridos mediante nota fiscal.",
    "O suporte pós-venda funciona exclusivamente de segunda a sexta, das 9h às 18h.",
    "O gerente geral da filial principal chama-se Carlos Souza e a loja fica na Av. Paulista, São Paulo.",
    "Para solicitar a troca de um produto com defeito de fábrica, o cliente tem até 90 dias conforme o CDC.",
]

vector_store = FAISS.from_texts(support_texts, embedding=embeddings)
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, "score_threshold": 0.65}
)

@tool
def search_bycode(code: str) -> str:
    """Busca um produto específico no banco de dados utilizando o código numérico do produto.
    Use esta ferramenta quando o usuário perguntar por um código (ex: 'qual o produto 1123?').
    """
    for produto in database:
        if produto["code"] == code:
            return f"O código informado está relacionado ao produto {produto['name']}, preço: {produto['price']}"
    
    return f"Não foi encontrado nenhum produto com o código {code}."

@tool
def search_byname(termo_busca: str) -> str:
    """Busca produtos no banco de dados cujo nome contenha o termo pesquisado.
    Use esta ferramenta quando o usuário perguntar se vocês vendem algo ou buscar pelo nome (ex: 'vocês vendem maçã?').
    """
    print("search_byname")
    termo_limpo = termo_busca.lower()
    encontrados = [
        p for p in database if termo_limpo in p["name"].lower()
    ]
    print(f"Temos {encontrados}")
    
    if not encontrados:
        return f"Não temos este produto disponível, posso ajudar com alguma outra coisa?"

    resultados_formatados = []
    for p in encontrados:
        resultados_formatados.append(f"{p['name']} {p['code']}: {p['price']}")
        
    return "Nós temos:\n" + "\n".join(resultados_formatados)

@tool
def convert_currency(amount: float, source_currency: str, target_currency: str) -> str:
    """Útil para converter valores de produtos entre moedas como USD, BRL e EUR."""
    rates = {
        ("USD", "BRL"): 5.50,
        ("BRL", "USD"): 0.18,
        ("EUR", "BRL"): 6.00,
        ("BRL", "EUR"): 0.17,
    }
    pair = (source_currency.upper(), target_currency.upper())
    if pair in rates:
        result = amount * rates[pair]
        return f"O valor de {amount} {source_currency} equivale a {result:.2f} {target_currency} na cotação da loja."
    return f"Não consegui converter de {source_currency} para {target_currency}."

@tool
def search_rules(query: str) -> str:
    """Útil para responder perguntas sobre regras da loja, políticas de reembolso, trocas, gerência e suporte."""
    try:
        documents = retriever.invoke(query)
        if not documents:
            return "Nenhuma regra específica encontrada na base de dados sobre este tema."
        found_texts = [doc.page_content for doc in documents]
        return "\n\n".join(found_texts)
    except:
        return "Nenhuma regra específica encontrada na base de dados sobre este tema."


class TriageSchema(BaseModel):
    reasoning: str = Field(description="Step-by-step decision making analyzing the recent history")
    classification: Literal["generalist", "specialist"] = Field(description="The chosen final route destination")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]         
    chosen_route: str
    encrypted_cpf: bytes


economic_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",  
    google_api_key=api_key
)

advanced_model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",    
    temperature=0.2,
    google_api_key=api_key
)

structured_triage_model = economic_model.with_structured_output(TriageSchema)

def security_node(state: AgentState):
    last_message = state["messages"][-1].content
    cpf_pattern = r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b'
    found_cpf = re.search(cpf_pattern, last_message)
    
    if found_cpf:
        clean_cpf = re.sub(r'\D', '', found_cpf.group(0))
        secure_cpf = cipher_suite.encrypt(clean_cpf.encode())
        masked_message = re.sub(cpf_pattern, "[CONFIDENTIAL_DATA_HIDDEN]", last_message)
        state["messages"][-1].content = masked_message
        return {"messages": state["messages"], "encrypted_cpf": secure_cpf}
    return {}


def triage_node(state: AgentState):
    history_text = ""
    for msg in state["messages"]:
        role = "Client" if msg.type == "human" else "Support"
        history_text += f"{role}: {msg.content}\n"
        
    triage_prompt = (
        "Você é o roteador de mensagens de time de suporte. Sua única função é analisar "
        "o histórico da conversa fornecido e classificar a última intenção do cliente.\n\n"
        "DIRETRIZES DE DECISÃO\n"
        "1. Escolha 'specialist' se o cliente quiser saber sobre políticas internas, suporte, regras da loja ou gerência.\n"
        "2. Escolha 'generalist' para produtos, preços, códigos, conversões de moedas ou conversas informais.\n\n"
        f"HISTÓRICO ATUAL PARA ANÁLISE\n{history_text}"
    )

    print("Triage Prompt:", triage_prompt)
    
    result = structured_triage_model.invoke(triage_prompt)
    return {"chosen_route": result.classification}

generalist_tools = [search_bycode, search_byname, convert_currency]
economic_model_with_tools = economic_model.bind_tools(generalist_tools)

def generalist_node(state: AgentState):
    try:
        response = economic_model_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
      print("ERRO:",e)

def specialist_node(state: AgentState):
    last_user_message = state["messages"][-1].content
    store_context = search_rules.invoke(last_user_message)
    
    contextualized_prompt = (
        "Você é o especialista de Atendimento Avançado da loja, seu dever é responder a dúvida do cliente "
        "utilizando estritamente as regras e dados fornecidos no contexto abaixo. Se o contexto "
        "informar que não possui dados, explique pacientemente que a loja não atende a essa solicitação específica.\n\n"
        f"--- CONTEXTO OFICIAL DA LOJA ---\n{store_context}\n\n"
        f"Pergunta do Cliente: {last_user_message}"
    )
    
    response = advanced_model.invoke(contextualized_prompt)
    return {"messages": [response]}


def conditional_router(state: AgentState) -> Literal["generalist", "specialist"]:
    if state.get("chosen_route") == "specialist":
        return "specialist"
    return "generalist"


builder = StateGraph(AgentState)

builder.add_node("security", security_node)
builder.add_node("triage", triage_node)
builder.add_node("generalist", generalist_node)
builder.add_node("specialist", specialist_node)
builder.add_node("tools_generalist", ToolNode(generalist_tools))

builder.add_edge(START, "security")
builder.add_edge("security", "triage")

builder.add_conditional_edges(
    "triage",
    conditional_router,
    {
        "generalist": "generalist",
        "specialist": "specialist"
    }
)

builder.add_conditional_edges(
    "generalist", 
    tools_condition, 
    {"tools": "tools_generalist", END: END}
)
builder.add_edge("tools_generalist", "generalist") 

builder.add_edge("generalist", END)
builder.add_edge("specialist", END)

local_memory = MemorySaver()
final_agent = builder.compile(checkpointer=local_memory)


class MessageSchema(BaseModel):
    chat_id: str
    texto: str


@app.post("/chat")
async def chat_endpoint(payload: MessageSchema):
    try:
        config = {"configurable": {"thread_id": payload.chat_id}}
       
        result = final_agent.invoke(
            {"messages": [HumanMessage(content=payload.texto)]}, 
            config=config
        )

        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(status_code=500, detail="No messages returned from agent.")
            
        final_response = messages[-1].content
        return {"resposta": final_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)