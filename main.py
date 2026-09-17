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


# Legacy alias: POST /upload -> 307 vers POST /documents (compat frontend ancien)
from fastapi import UploadFile, File, HTTPException, status
from typing import Annotated

FileUpload = Annotated[UploadFile, File()]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 Mo legacy


@app.post("/upload", deprecated=True, include_in_schema=False)
async def upload_document_legacy(file: FileUpload):
    """Legacy POST /upload — redirige vers /documents pipeline.
    Gardé pour compat, mais /documents est la route recommandée (multipart PDF).
    """
    # Délègue à documents router en important la fonction
    from routers.documents import upload_document as doc_upload

    return await doc_upload(file)