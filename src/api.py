import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_traffic


app = FastAPI(
    title="Hybrid Network Traffic Analysis API",
    description=(
        "Network anomaly detection using "
        "Random Forest and Isolation Forest."
    ),
    version="2.0.0",
)


class TrafficRequest(BaseModel):
    records: list[dict]


@app.get("/")
def home():

    return {
        "message": "Hybrid Network Traffic Analysis API",
        "algorithms": [
            "Random Forest",
            "Isolation Forest",
        ],
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

        df = pd.DataFrame(
            request.records
        )

        results = predict_traffic(df)

        return {
            "total_flows": len(results),

            "normal_flows": int(
                (
                    results["status"] == "Normal"
                ).sum()
            ),

            "potential_anomalies": int(
                (
                    results["status"]
                    == "Potential Anomaly"
                ).sum()
            ),

            "selected_strategy": (
                results["selected_strategy"].iloc[0]
            ),

            "results": results.to_dict(
                orient="records"
            ),
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