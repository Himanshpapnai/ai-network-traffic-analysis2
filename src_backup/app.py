"""NetSentinels dashboard server.  Run:  python app.py   (or double-click run_dashboard.bat)"""
import threading, webbrowser
from pathlib import Path
import numpy as np, pandas as pd
from flask import Flask, jsonify, send_from_directory

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
app = Flask(__name__, static_folder=None)

def newest_csv():
    files = sorted(DATA.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def to_rows(df):
    df.columns = [c.strip() for c in df.columns]
    def num(name):
        if name not in df:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    rf = num("rf_attack_probability") if "rf_attack_probability" in df else num("rf_prediction")
    iff = num("if_anomaly_score") if "if_anomaly_score" in df else num("if_prediction")
    out = pd.DataFrame({
        "port": num("Destination Port"), "dur": num("Flow Duration"),
        "fp": num("Total Fwd Packets"), "bp": num("Total Backward Packets"),
        "lf": num("Total Length of Fwd Packets"), "lb": num("Total Length of Bwd Packets"),
        "rfp": rf, "ifs": iff,
        "lab": df["Label"].astype(str).str.strip() if "Label" in df else "BENIGN"})
    return out.head(200000).to_dict("records")

@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")

@app.get("/api/results")
def results():
    p = newest_csv()
    if p is None:
        return jsonify(error="Put a results CSV in the data folder."), 404
    return jsonify(rows=to_rows(pd.read_csv(p)), source=p.name)

@app.post("/api/predict")
def predict():
    return jsonify(error="This CSV has no model-result columns. Upload the output of predict.py "
                         "(needs rf_attack_probability and if_anomaly_score), or copy it into the data folder."), 400

@app.get("/favicon.ico")
def favicon():
    return "", 204
if __name__ == "__main__":
    print("Using results file:", newest_csv())
    print("Open http://127.0.0.1:5050  (Ctrl+C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5050")).start()
    import logging; logging.getLogger("werkzeug").setLevel(logging.ERROR)
app.run(host="127.0.0.1", port=5050)
