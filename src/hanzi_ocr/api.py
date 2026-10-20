import os
from contextlib import asynccontextmanager
from tempfile import mkdtemp

import boto3
from fastapi import FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel

from hanzi_ocr.model_inference import (
    create_character_transformers,
    inference,
    load_model,
    preprocess_image,
)
from hanzi_ocr.model_storage import download_model


class Prediction(BaseModel):
    probability: float
    simplified: str
    traditional: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = os.environ.get("DEVICE", "cpu")
    model_dir = os.environ.get("MODEL_DIR")

    if model_dir is None:
        bucket = os.environ["MODEL_BUCKET"]
        s3_client = boto3.client("s3")
        model_name = os.environ.get("MODEL_NAME", "hanziOCR").strip("/", "")
        model_version = os.environ.get("MODEL_VERSION", "V1").strip("/", "")

        prefix = f"{model_name}/{model_version}/"
        model_dir = download_model(s3_client, bucket, prefix, dest_dir=mkdtemp())

    app.state.model, app.state.le = load_model(model_dir, device)
    app.state.device = device
    app.state.simplified_converter, app.state.traditional_converter = (
        create_character_transformers()
    )

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=list[Prediction])
def predict(request: Request, file: UploadFile, top_k: int = 5):
    image_bytes = file.file.read()
    try:
        image = preprocess_image(image_bytes).to(request.app.state.device)
    except ValueError:
        raise HTTPException(status_code=400, detail="File could not be parsed as image")
    try:
        predictions_with_proba = inference(
            request.app.state.model,
            request.app.state.le,
            image,
            top_k,
            request.app.state.simplified_converter,
            request.app.state.traditional_converter,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    predictions = [
        Prediction(probability=proba, simplified=simplified, traditional=traditional)
        for proba, (simplified, traditional) in predictions_with_proba
    ]
    return predictions
