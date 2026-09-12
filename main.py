from typing import Annotated
from fastapi import FastAPI, UploadFile, File, HTTPException, status

app = FastAPI()
FileUpload = Annotated[UploadFile, File()]
MAX_FILE_SIZE = 5 * 1024 * 1024 # 5 Mo

@app.get("/")
def root():
    return {"message": "Hello from Folio"}

@app.get("/health")
def health():
    return {
        "message": "it's working",
        "status": 200
    }

@app.post("/upload")
async def upload_document(file: FileUpload):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )
    
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="No more than 5 Mo"
        )

    await file.seek(0) # Pourquoi ceci !?

    return {
        "message": f"Le document {file.filename} a bien été reçu",
        "filename": file.filename,
        "file_size": len(contents),
        "content_type": file.content_type
    }


"""
## Correction à faire plutard:
Deux points à garder en tête pour l'étape "durcissement" plus tard (pas bloquant maintenant) :

file.content_type est déclaré par le client, donc falsifiable — quelqu'un peut envoyer n'importe quoi en mettant Content-Type: application/pdf. Pour une vraie validation, il faudra vérifier les premiers octets du fichier (%PDF-) en plus du header.
Tu lis contents = await file.read() (tout en mémoire) avant de vérifier la taille — un fichier de 2 Go étiqueté PDF serait entièrement chargé en RAM avant d'être rejeté. Pour la démo publique, il vaudra mieux vérifier Content-Length ou lire par chunks avec arrêt anticipé.
"""