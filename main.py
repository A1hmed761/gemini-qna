from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

from rag import process_file, ask_question

app = FastAPI()

# allow Firebase fronten
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your Firebase domain
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # delete old FAISS
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

    process_file(file_path)

    return {"message": "File processed successfully"}


from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str


@app.post("/ask")
async def ask(request: QueryRequest):
    answer = ask_question(request.query)
    return {"answer": answer}