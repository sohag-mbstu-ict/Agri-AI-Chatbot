import os
import time
import logging
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

# LLM
from langchain_groq import ChatGroq

# Tools
from langchain_tavily import TavilySearch
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun

# Agent (MODERN LANGCHAIN)
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler


# -----------------------------
# 1. ENV LOAD
# -----------------------------
env_path = Path("/home/gflml/Chatbot/.env")
load_dotenv(dotenv_path=env_path)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or os.getenv("GFL_TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY API KEY NOT FOUND")
if not GROQ_API_KEY:
    raise ValueError("GROQ API KEY NOT FOUND")


# -----------------------------
# 2. LOGGING SETUP
# -----------------------------
logging.basicConfig(
    filename="agent_tool_trace.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


def log_file(msg: str):
    logging.info(msg)


# -----------------------------
# 3. TOOL CALLBACK (TRACING)
# -----------------------------
class ToolLogger(BaseCallbackHandler):

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.start_time = time.time()

        tool_name = serialized.get("name", "unknown")

        print("\n🟡 TOOL START")
        print(f"Tool: {tool_name}")
        print(f"Input: {input_str}")

        log_file(f"START | {tool_name} | {input_str}")

    def on_tool_end(self, output, **kwargs):
        duration = time.time() - self.start_time

        print("\n🟢 TOOL END")
        print(f"Output: {str(output)[:300]}")
        print(f"Time: {duration:.2f}s")

        log_file(f"END | OUTPUT | {str(output)[:300]} | {duration:.2f}s")

    def on_tool_error(self, error, **kwargs):
        print("\n🔴 TOOL ERROR")
        print(error)

        log_file(f"ERROR | {error}")


# -----------------------------
# 4. LLM LOADER
# -----------------------------
@lru_cache(maxsize=1)
def load_llm(model_name="qwen/qwen3-32b", temperature=0):
    print("🔹 Loading Groq LLM...")
    return ChatGroq(
        temperature=temperature,
        groq_api_key=GROQ_API_KEY,
        model_name=model_name
    )


# -----------------------------
# 5. TOOLS
# -----------------------------
def load_tools():
    tavily = TavilySearch(
        tavily_api_key=TAVILY_API_KEY,
        max_results=5,
        topic="general"
    )

    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=500
        )
    )

    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=500
        )
    )

    return [tavily, wiki, arxiv]


# -----------------------------
# 6. BUILD AGENT
# -----------------------------
@lru_cache(maxsize=1)
def build_agent():
    llm = load_llm()
    tools = load_tools()

    print("🔹 Building Tool Calling Agent...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use tools when required."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        callbacks=[ToolLogger()],
        handle_parsing_errors=True,
        return_intermediate_steps=True   # IMPORTANT
    )


# -----------------------------
# 7. ASK FUNCTION (WITH FULL TRACE)
# -----------------------------
def ask_agent(query: str):
    agent_executor = build_agent()
    response = agent_executor.invoke({"input": query})
    steps = response.get("intermediate_steps", [])
    print("\n\n================ FULL TOOL TRACE ================\n")
    for i, step in enumerate(steps):
        action = step[0]
        result = step[1]
        tool_name = getattr(action, "tool", "unknown")
        tool_input = getattr(action, "tool_input", {})
        print(f"\n🔹 STEP {i+1}")
        print(f"Tool: {tool_name}")
        print(f"Input: {tool_input}")
        print(f"Output: {str(result)[:300]}")
        log_file(f"STEP {i+1} | {tool_name} | {tool_input} | {str(result)[:300]}")
    print("\n================================================\n")
    return response["output"]



# import streamlit as st
# # -----------------------------
# # STREAMLIT CONFIG
# # -----------------------------
# st.set_page_config(page_title="Agri AI Chatbot", layout="wide")

# st.title("🌾 Dr. Chashi AI Assistant")
# st.write("Ask agricultural questions in Bangla or English")

# # -----------------------------
# # SESSION STATE (chat history)
# # -----------------------------
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # -----------------------------
# # USER INPUT
# # -----------------------------
# query = st.text_input("🔍 Enter your question:")

# col1, col2 = st.columns([1,1])

# with col1:
#     ask_btn = st.button("Ask")

# with col2:
#     clear_btn = st.button("Clear Chat")

# # -----------------------------
# # CLEAR CHAT
# # -----------------------------
# if clear_btn:
#     st.session_state.chat_history = []
#     st.rerun()

# # -----------------------------
# # ASK AGENT
# # -----------------------------
# if ask_btn:
#     if not query.strip():
#         st.warning("Please enter a question")
#     else:
#         with st.spinner("Thinking..."):
#             result = ask_agent(query)

#         # Save history
#         st.session_state.chat_history.append({
#             "query": query,
#             "answer": result
#         })

# # -----------------------------
# # DISPLAY CHAT
# # -----------------------------
# for chat in reversed(st.session_state.chat_history):
#     st.markdown(f"### 🧑 User:\n{chat['query']}")
#     st.markdown(f"### 🤖 AI:\n{chat['answer']}")
#     st.markdown("---")

# # # -----------------------------
# # # 8. CLI LOOP
# # # -----------------------------
# # if __name__ == "__main__":
# #     while True:
# #         query = input("Enter your input (b to break): ")
# #         if query.lower() == "b":
# #             break
# #         result = ask_agent(query)
# #         print("\n✅ FINAL ANSWER:\n", result)