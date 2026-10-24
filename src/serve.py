import logging
import time

import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import MLP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

model = MLP(hidden=cfg["hidden_size"])
model.load_state_dict(torch.load(cfg["model_path"]))
model.eval()

app = FastAPI()


class Item(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(item: Item):
    if len(item.features) != 64:
        raise HTTPException(status_code=422, detail="need 64 features")

    start = time.time()
    x = torch.tensor([item.features], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()

    logger.info("prediction=%s latency_ms=%.1f", pred, (time.time() - start) * 1000)
    return {"prediction": pred}
