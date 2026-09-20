from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.documents import router as documents_router
from routers.query import router as query_router

app = FastAPI(title="Folio V0", version="0.5.0")

# CORS pour Vite dev server (frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router, prefix="/documents", tags=["documents"])
app.include_router(query_router, prefix="/query", tags=["query"])


@app.get("/")
def root():
    return {"message": "Hello from Folio", "version": "0.5.0"}


@app.get("/health")
def health():
    return {"message": "it's working", "status": 200}