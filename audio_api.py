import io
import librosa
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response

app = FastAPI()

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

    y, sr = librosa.load(io.BytesIO(contents), sr=None)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.array(tempo).flatten()[0])

    fig = plt.figure(figsize=(6,4))
    plt.plot(y[:5000])
    plt.title(f"BPM: {round(tempo,1)}")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="image/png"
    )
