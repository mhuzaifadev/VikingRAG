from vikingrag.api.routers.answers import router as answers_router
from vikingrag.api.routers.documents import router as documents_router
from vikingrag.api.routers.health import router as health_router
from vikingrag.api.routers.retrieval import router as retrieval_router
from vikingrag.api.routers.search import router as search_router

__all__ = [
    "answers_router",
    "documents_router",
    "health_router",
    "retrieval_router",
    "search_router",
]
