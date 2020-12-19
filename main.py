import io
from fastapi import FastAPI, File, UploadFile
import faceEncode

app = FastAPI()

@app.post("/uploadfile/")
async def get_img_embed(file: UploadFile = File(...)):
    contents = await file.read()
    contents = io.BytesIO(contents)
    embed = faceEncode.main(contents)
    return embed