import io, json
from fastapi import FastAPI, UploadFile, File
from model import Shapes
from PIL import Image
import torch
from torchvision import transforms


app = FastAPI()

with open("classes(1).json", "r", encoding="utf-8") as f:
    classes = json.load(f)

model = Shapes()
model.load_state_dict(torch.load("model(1).pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def prediction(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        idx = model(tensor).argmax(1).item()
    return {"class": classes[idx]}
