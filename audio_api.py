import matplotlib
matplotlib.use("Agg")

import io
import librosa
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response

app = FastAPI()

@app.get("/")
def root():
    return {"status": "alive"}

@app.post(
    "/analyze/",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "PNG image"
        }
    }
)
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()

    # 🔥 On limite fortement la charge
    y, sr = librosa.load(
        io.BytesIO(contents),
        sr=22050,
        mono=True,
        duration=30   # max 30 secondes
    )

    # 🔥 Tempo simplifié (plus léger)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    fig = plt.figure(figsize=(6,4))
    plt.plot(y[::30])
    plt.title(f"BPM: {round(float(tempo),1)}")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="image/png"
    )
