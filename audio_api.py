import matplotlib
matplotlib.use("Agg")

import io
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import soundfile as sf

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

        # 🔥 Lire seulement 15 secondes max
        data, samplerate = sf.read(io.BytesIO(contents))
        max_samples = samplerate * 15
        data = data[:max_samples]

        # 🔥 Si stéréo → mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # 🔥 Estimation BPM ultra simple (basée sur pics RMS)
        frame_size = 1024
        energy = [
            np.sum(np.abs(data[i:i+frame_size]))
            for i in range(0, len(data), frame_size)
        ]

        energy = np.array(energy)
        peaks = np.where(energy > np.mean(energy))[0]

        if len(peaks) > 1:
            intervals = np.diff(peaks)
            avg_interval = np.mean(intervals)
            bpm = 60 * samplerate / (avg_interval * frame_size)
        else:
            bpm = 0

        bpm = round(float(bpm), 1)

        # 🔥 Graph léger
        fig = plt.figure(figsize=(6,4))
        plt.plot(data[:3000])
        plt.title(f"BPM approx: {bpm}")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)

        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
