from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,GoogleSearchRun
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from langgraph.graph import END, StateGraph, START
from langchain_groq import ChatGroq
from langchain.schema import Document
from typing import List
from typing_extensions import TypedDict
import streamlit as st
import cassio
import os
from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID=os.getenv("ASTRA_DB_ID")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_CSE_ID"] = "15929e2e83437452b"
os.environ["GOOGLE_API_KEY"] = "AIzaSyC1yMjtCSLcCf5OxrxXMx7llkfBHBSH9YU"

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

@st.cache_resource
def initialize_resources():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Cassandra(
        embedding=embeddings,
        table_name="qa_mini_demo",
        session=None,
        keyspace=None
    )
    return vector_store


docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

astra_vector_store=initialize_resources()

astra_vector_store.add_documents(doc_splits)

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
retriever=astra_vector_store.as_retriever()
retriever.invoke("What is agent",ConsistencyLevel="LOCAL_ONE")

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search","google_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore or google search.",
    )

groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or wikipedia or google search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search and if needed use google search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

search = GoogleSearchAPIWrapper()
google = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):
    """
    wiki search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---wikipedia---")
    print("---HELLO--")
    question = state["question"]
    print(question)

    docs = wiki.invoke({"query": question})
    
    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)

    return {"documents": wiki_results, "question": question}

def google_search(state):
    """
    google search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---Google---")
    print("---HELLO--")
    question = state["question"]
    print(question)

    docs = google.run({"query": question})
    google_results = docs
    google_results = Document(page_content=google_results)

    return {"documents": google_results, "question": question}

def route_question(state):
    """
    Route question to wiki search, vectorstore, or google search.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    elif source.datasource == "google_search":
        print("---ROUTE QUESTION TO Google Search---")
        return "google_search"


workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)  
workflow.add_node("google_search", google_search)
workflow.add_node("retrieve", retrieve) 
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
        "google_search": "google_search",
    },
)
workflow.add_edge( "retrieve", END)
workflow.add_edge( "wiki_search", END)
workflow.add_edge( "google_search", END)

st_app = workflow.compile()

st.title("StateGraph Query App")
st.write("Ask a question and get results from the best-suited data source.")


user_question = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_question:
        st.write("Processing your question...")
        
        inputs = {"question": user_question}
        result_docs = None

        for output in st_app.stream(inputs):
            for key, value in output.items():
                st.write(f"Node '{key}':")
            result_docs = value.get('documents', None)

        if result_docs:
            st.write("### Results:")

            if isinstance(result_docs, Document):
                st.write(result_docs.page_content)
            elif isinstance(result_docs, list):
                for doc in result_docs:
                    st.write(doc.page_content)
            else:
                st.write("Unexpected result format.")
        else:
            st.write("No results found.")
    else:
        st.error("Please enter a question to proceed.")
