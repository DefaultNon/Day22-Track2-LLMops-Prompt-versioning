import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "day22-lab")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langsmith import Client, traceable

from config import load_config
from step1_langsmith_rag_pipeline import build_vectorstore, SAMPLE_QUESTIONS

config = load_config()

# Prompt Hub names (unique to user)
PROMPT_V1_NAME = f"rag-prompt-v1-{os.getenv('USER', 'default')}"
PROMPT_V2_NAME = f"rag-prompt-v2-{os.getenv('USER', 'default')}"

# Define templates
SYSTEM_V1 = (
    "You are a strict factual assistant. Answer the user question using ONLY the provided context. "
    "Rules:\n"
    "1. Every sentence in your response MUST be directly traceable to a specific sentence in the context.\n"
    "2. If the exact information is not present, say 'I don't have enough information.'\n"
    "3. Do not add outside knowledge, adjectives, or logical leaps.\n"
    "4. Keep it very concise.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured answer based EXCLUSIVELY on the provided context.\n\n"
    "Strict Grounding Rules:\n"
    "1. Before answering, extract the relevant sentences from the context.\n"
    "2. Ensure the answer contains ONLY facts found in those sentences.\n"
    "3. If the context does not contain the answer, explicitly state 'Insufficient context to answer'.\n"
    "4. NO outside information or hallucinations are permitted.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])

def push_prompts_to_hub(client):
    """Push prompts to LangSmith Prompt Hub."""
    try:
        url1 = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – concise answers")
        print(f"[OK] Pushed V1 -> {url1}")
    except Exception as e:
        print(f"[WARN] V1: {e}")

    try:
        url2 = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – structured answers")
        print(f"[OK] Pushed V2 -> {url2}")
    except Exception as e:
        print(f"[WARN] V2: {e}")

def pull_prompts_from_hub(client):
    """Pull prompts from Hub with local fallback."""
    prompts = {}
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"[PULL] Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"[INFO] Using local fallback for V1")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"[PULL] Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"[INFO] Using local fallback for V2")
    
    return prompts

def get_prompt_version(request_id: str) -> str:
    """Deterministic A/B routing based on request_id hash."""
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """Run RAG with specific prompt version."""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return {"question": question, "answer": answer, "version": version}

def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)
    
    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
    
    # Push prompts
    push_prompts_to_hub(client)
    
    # Pull prompts
    prompts = pull_prompts_from_hub(client)
    
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOpenAI(
        model=config["default_llm"],
        api_key=config["api_key"],
        base_url=config["openai_endpoint"],
        temperature=0,  # Ensure deterministic and faithful outputs
    )
    
    v1_count = 0
    v2_count = 0
    
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]
        
        if version_tag == "v1": v1_count += 1
        else: v2_count += 1
        
        result = ask_ab(retriever, llm, prompt, question, version_tag)
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")
    
    print("-" * 30)
    print(f"Routing Summary: V1={v1_count}, V2={v2_count}")
    print(f"[OK] Total {len(SAMPLE_QUESTIONS)} queries processed.")

if __name__ == "__main__":
    main()
