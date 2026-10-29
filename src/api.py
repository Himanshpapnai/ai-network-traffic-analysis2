from typing import Any

import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_traffic


app = FastAPI(
    title="AI Network Traffic Analysis API",
    description=(
        "Detect unusual network flows using Isolation Forest."
    ),
    version="1.0.0",
)


class TrafficRequest(BaseModel):
    records: list[dict[str, Any]]


@app.get("/")
def home():
    return {
        "message": "AI Network Traffic Analysis API is running",
        "algorithm": "Isolation Forest",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/predict")
def predict(request: TrafficRequest):
    if not request.records:
        raise HTTPException(
            status_code=400,
            detail="At least one traffic record is required.",
        )

    try:
        df = pd.DataFrame(request.records)

        results = predict_traffic(df)

        return {
            "total_flows": len(results),
            "normal_flows": int(
                (results["status"] == "Normal").sum()
            ),
            "anomalous_flows": int(
                (
                    results["status"] == "Potential Anomaly"
                ).sum()
            ),
            "results": results.to_dict(orient="records"),
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc