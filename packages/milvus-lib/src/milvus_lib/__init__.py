import itertools
import logging
from typing import Any, Dict, Iterator, List

from pydantic import BaseModel

# PyMilvus imports
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    connections,
    utility,
)

logger = logging.getLogger(__name__)


class SearchComponent(BaseModel):
    title: str
    url: str
    description: str
    parent: str
    accessibility: str
    created_at: str
    updated_at: str
    has_research: bool
    views: int


class ScoredSearchComponent(SearchComponent):
    score: float


class ComponentEntry(BaseModel):
    component_id: str
    title: str
    description: str
    url: str
    parent: str
    accessibility: str
    has_research: bool
    created_at: str
    updated_at: str
    views: int
    content: str
    full_content: str


class MilvusKnowledgeBase:
    host: str
    port: int
    collection_name: str
    embedding_model: str
    embedding_dim: int
    max_batch_size: int
    collection: Collection

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "knowledge_base",
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
        embedding_dim: int = 1024,
        max_batch_size: int = 2,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.max_batch_size = max_batch_size

    def __schema(self) -> CollectionSchema:
        return CollectionSchema(
            fields=self.__fields(),
            functions=self.__functions(),
            description="Knowledge base for design system component",
        )

    def __index(self) -> Dict[str, Any]:
        return {
            "metric_type": "COSINE",  # Use cosine similarity
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }

    def __fields(self) -> List[FieldSchema]:
        return [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="Primary key",
            ),
            FieldSchema(
                name="component_id",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="Unique component identifier",
            ),
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Component title",
            ),
            FieldSchema(
                name="description",
                dtype=DataType.VARCHAR,
                max_length=4096,
                description="Component description",
            ),
            FieldSchema(
                name="url",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Component URL",
            ),
            FieldSchema(
                name="parent",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="Parent design system",
            ),
            FieldSchema(
                name="accessibility",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="Accessibility level (e.g., AA)",
            ),
            FieldSchema(
                name="has_research",
                dtype=DataType.BOOL,
                description="Whether component has research",
            ),
            FieldSchema(
                name="created_at",
                dtype=DataType.VARCHAR,
                max_length=128,
                description="Creation timestamp",
            ),
            FieldSchema(
                name="updated_at",
                dtype=DataType.VARCHAR,
                max_length=128,
                description="Update timestamp",
            ),
            FieldSchema(name="views", dtype=DataType.INT16, description="views"),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Markdown content to be embedded",
            ),
            FieldSchema(
                name="content_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
                description="Vector embedding of component content",
            ),
            FieldSchema(
                name="full_content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Full markdown content (truncated if needed)",
            ),
        ]

    def __functions(self) -> List[Function]:
        params = {  # Provider-specific configuration (highest priority)
            "provider": "openai",  # Embedding model provider
            "model_name": self.embedding_model,  # Embedding model
            "dim": self.embedding_dim,
            # Optional parameters:
            # "credential": "apikey_dev",               # Optional: Credential label specified in milvus.yaml
            # "user": "user123"                         # Optional: identifier for API tracking
        }
        logger.warning(f"Embedding model parameters: {params}")
        return [
            Function(
                name="content_embedding",  # Unique identifier for this embedding function
                function_type=FunctionType.TEXTEMBEDDING,  # Type of embedding function
                input_field_names=["content"],  # Scalar field to embed
                output_field_names=[
                    "content_embedding"
                ],  # Vector field to store embeddings
                params=params,
            ),
        ]

    def is_healthy(self) -> bool:
        return self.collection is not None

    def connect(self, drop_existing: bool = False):
        connections.connect(alias="default", host=self.host, port=self.port)

        if drop_existing and utility.has_collection(self.collection_name):
            logger.info(f"Dropping existing collection: {self.collection_name}")
            utility.drop_collection(self.collection_name)

        # Get collection and load it
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            logger.info(f"Creating new collection: {self.collection_name}")
            self.collection = Collection(self.collection_name, schema=self.__schema())
            self.collection.create_index(
                field_name="content_embedding", index_params=self.__index()
            )

        self.collection.load()
        logger.info(f"Connected to Milvus collection: {self.collection_name}")

    def close(self):
        self.collection.release()
        self.collection = None
        connections.disconnect("default")
        logger.info(f"Disconnected from Milvus collection: {self.collection_name}")

    def add_components(self, components: Iterator[ComponentEntry]):

        for batch in itertools.batched(
            map(lambda x: x.model_dump(), components), self.max_batch_size
        ):
            self.collection.insert([*batch])
        self.collection.flush()

    def search_components(
        self, query: List[float] | str, limit: int = 10
    ) -> List[ScoredSearchComponent]:
        # Search parameters
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        # Perform search
        results = self.collection.search(
            data=[query],
            anns_field="content_embedding",
            param=search_params,
            limit=limit,
            output_fields=[
                "title",
                "description",
                "url",
                "parent",
                "accessibility",
                "has_research",
                "created_at",
                "updated_at",
                "views",
            ],
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = ScoredSearchComponent(
                    score=hit.score,
                    title=hit.entity.get("title"),
                    description=hit.entity.get("description"),
                    url=hit.entity.get("url"),
                    parent=hit.entity.get("parent"),
                    accessibility=hit.entity.get("accessibility"),
                    has_research=hit.entity.get("has_research"),
                    created_at=hit.entity.get("created_at"),
                    updated_at=hit.entity.get("updated_at"),
                    views=hit.entity.get("views"),
                )
                formatted_results.append(result)
        return formatted_results
