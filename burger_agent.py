import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langfuse.langchain import CallbackHandler

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

MENU_DATA = [
    "Big Mac: $5",
    "Whopper: $6", 
    "Fries: $2",
]

vectorstore = FAISS.from_texts(texts=MENU_DATA, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def lookup_price(query: str) -> str:
    """
    Search the menu database for item prices.
    Use this tool when the user asks about menu items or prices.
    
    Args:
        query: The search query for menu items (e.g., "Big Mac price")
    
    Returns:
        Relevant menu items and their prices
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No items found matching your query."
    return "\n".join([doc.page_content for doc in docs])


@tool
def place_order(items: str) -> str:
    """
    Place an order for the specified items.
    Use this tool when the user wants to order items from the menu.
    
    Args:
        items: Comma-separated list of items to order (e.g., "Big Mac, Fries")
    
    Returns:
        Order confirmation message
    """
    return f"ORDER_PLACED: [{items}]"


tools = [lookup_price, place_order]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)

system_prompt = """You are a helpful burger shop assistant. Your job is to:
1. Help customers find menu items and prices using the lookup_price tool
2. Place orders for customers using the place_order tool

Always be friendly and helpful. When a customer asks about prices, use the lookup_price tool first.
When they want to order, use the place_order tool with the items they requested.

Important: Always look up prices before confirming them to customers."""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    debug=False,
)


def run_agent(user_input: str) -> tuple[str, str | None]:
    """
    Run the burger shop agent with the given user input.
    Includes Langfuse observability via callback handler.
    
    Args:
        user_input: The customer's message/query
        
    Returns:
        Tuple of (agent_response, trace_id)
    """
    langfuse_handler = CallbackHandler()
    inputs = {"messages": [HumanMessage(content=user_input)]}
    config = {"callbacks": [langfuse_handler]}
    result = agent.invoke(inputs, config=config)
    trace_id = getattr(langfuse_handler, 'last_trace_id', None)
    
    messages = result.get("messages", [])
    if messages:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return (str(msg.content), trace_id)
            elif hasattr(msg, "content") and msg.content and not isinstance(msg, HumanMessage):
                return (str(msg.content), trace_id)
    
    return (str(result), trace_id)


if __name__ == "__main__":
    print("=" * 50)
    print("Burger Shop Agent - Test Run")
    print("=" * 50)
    
    test_query = "How much is a Big Mac and please order one."
    print(f"\nCustomer: {test_query}\n")
    
    response, trace_id = run_agent(test_query)
    print(f"\nAgent: {response}")
    if trace_id:
        print(f"Trace ID: {trace_id}")

