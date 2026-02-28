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

    try:
        contents = await file.read()

        # 🔥 On limite la taille lue (sécurité RAM)
        max_bytes = 5 * 1024 * 1024  # 5MB max analysés
        contents = contents[:max_bytes]

        # 🔥 On charge seulement 20 secondes max
        y, sr = librosa.load(
            io.BytesIO(contents),
            sr=22050,
            duration=20
        )

        # 🔥 On downsample encore pour réduire la charge
        y = y[::5]

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.array(tempo).flatten()[0])

        fig = plt.figure(figsize=(6, 4))
        plt.plot(y[:3000])
        plt.title(f"BPM: {round(tempo, 1)}")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)

        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
