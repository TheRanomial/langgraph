from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.schema import Document
from typing import List, Union
from app import st_app


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    documents: Union[str, List[str]]


app = FastAPI()


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    inputs = {"question": request.question}
    result_docs = None
    for output in st_app.stream(inputs): 
        for _, value in output.items():
            result_docs = value.get("documents", None)

    if result_docs:
        if isinstance(result_docs, Document):
            response = {"documents": result_docs.page_content}
        elif isinstance(result_docs, list):
            response = {"documents": [doc.page_content for doc in result_docs]}
        else:
            response = {"documents": "Unexpected result format."}
        return response
    else:
        raise HTTPException(status_code=404, detail="No results found.")
