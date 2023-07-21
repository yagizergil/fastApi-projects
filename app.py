from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

# Depolama için kullanılacak geçici veritabanı
db: Dict[int, str] = {}

def division(a , b):
    return a/b

division(10 , 4)



# Metin verilerini temsil etmek için BaseModel sınıfını kullanıyoruz
class Text(BaseModel):
    content: str

# Yeni metin oluşturmak için POST isteği
@app.post("/texts/")
def create_text(text: Text):
    text_id = len(db) + 1
    db[text_id] = text.content
    return {"text_id": text_id}

# Belirli bir metni id ile almak için GET isteği
@app.get("/texts/{text_id}")
def get_text(text_id: int):
    if text_id not in db:
        return {"error": "Text bulunamadı."}
    return {"text_id": text_id, "content": db[text_id]}

@app.delete("/texts/{text_id}")
def delete_text(text_id: int):
    if text_id not in db:
        return{"error": "Text Bulunamadı."}
    return {"text : Text Silindi..."}

@app.get("/hello/{name}")
async def hello(name):
    return{"name" : name}

@app.put("/texts/{text_id}")
def put_text(text_id: int):
    if text_id not in db:
        return{"error" : "Text Bulunamadı."}
    return{"text_id": text_id, "content": db[text_id]}
