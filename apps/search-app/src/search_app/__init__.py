# Python imports
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_health import health
from milvus_lib import MilvusKnowledgeBase, SearchComponent
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

logger = logging.getLogger(f"uvicorn.{__name__}")


def create_knowledge_base() -> MilvusKnowledgeBase:
    """Create a MilvusKnowledgeBase from environment variables.

    Returns:
        A configured MilvusKnowledgeBase instance. Connection is established
        separately via lifespan.
    """
    host = os.getenv("MILVUS_HOST", "localhost")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    collection_name = os.getenv("MILVUS_COLLECTION", "knowledge_base")
    embedding_model = os.getenv(
        "MILVUS_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"
    )
    embedding_dim = int(os.getenv("MILVUS_EMBEDDING_DIM", "1024"))
    max_batch_size = int(os.getenv("MILVUS_MAX_BATCH_SIZE", "2"))
    return MilvusKnowledgeBase(
        host, port, collection_name, embedding_model, embedding_dim, max_batch_size
    )


knowledge_base = create_knowledge_base()


def knowledge_base_status() -> bool:
    """Check if the Milvus knowledge base is healthy and ready for search."""
    return knowledge_base.is_healthy()


async def health_handler(**kwargs) -> Dict[str, Any]:
    """Format health check results for the /health endpoint response."""
    is_success = all(kwargs.values())
    return {
        "status": "success" if is_success else "failure",
        "results": kwargs.items(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage knowledge base connection for the lifetime of the FastAPI app."""
    knowledge_base.connect()
    yield
    knowledge_base.close()


app = FastAPI(
    title="MOJ Design System Search", description="Vector Search API", lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """Root endpoint returning a simple greeting."""
    return {"Hello": "World"}


# Based on https://gist.github.com/Aron-v1/f6e58554acf9ef0f328ac93d74dcb9ca
class SearchRequest(BaseModel):
    """Request body for the /search endpoint."""

    message: str


class SearchResponse(BaseModel):
    """Response body for the /search endpoint."""

    message: str
    components: list[SearchComponent]


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform semantic search over design system components."""
    logger.info(f"Searching for: {request.message}")
    results = knowledge_base.search_components(request.message)
    logger.info(f"Search results: {results}")
    return SearchResponse(message="Search successful", components=results)
