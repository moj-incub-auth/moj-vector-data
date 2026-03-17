# Python imports
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi_health import health
from milvus_lib import MilvusKnowledgeBase, SearchComponent
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

logger = logging.getLogger(f"uvicorn.{__name__}")


def create_knowledge_base() -> MilvusKnowledgeBase:
    host = os.getenv("MILVUS_HOST", "localhost")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    collection_name = os.getenv("MILVUS_COLLECTION", "knowledge_base")
    return MilvusKnowledgeBase(host, port, collection_name)


knowledge_base = create_knowledge_base()


def knowledge_base_status() -> bool:
    return knowledge_base.is_healthy()


async def health_handler(**kwargs) -> Dict[str, Any]:
    is_success = all(kwargs.values())
    return {
        "status": "success" if is_success else "failure",
        "results": kwargs.items(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    knowledge_base.connect()
    yield
    knowledge_base.close()


app = FastAPI(
    title="MOJ Design System Search", description="Vector Search API", lifespan=lifespan
)

app.add_api_route(
    "/health",
    health(
        [knowledge_base_status],
        success_handler=health_handler,
        failure_handler=health_handler,
    ),
)
Instrumentator(
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app)


@app.get("/")
def read_root():
    return {"Hello": "World"}


# Based on https://gist.github.com/Aron-v1/f6e58554acf9ef0f328ac93d74dcb9ca
class SearchRequest(BaseModel):
    message: str


class SearchResponse(BaseModel):
    message: str
    components: list[SearchComponent]


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    results = knowledge_base.component_search(request.message)
    return SearchResponse(message="Search successful", components=results)
