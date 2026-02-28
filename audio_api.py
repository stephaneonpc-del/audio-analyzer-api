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


@app.post("/analyze/", response_class=Response)
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()
    data, samplerate = sf.read(io.BytesIO(contents))

    # 🔥 Limite 15 secondes
    max_samples = samplerate * 15
    data = data[:max_samples]

    # 🔥 Mono si stéréo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # -----------------------------
    # 🎵 ANALYSE SIMPLE ET LÉGÈRE
    # -----------------------------

    # BPM approx via énergie
    frame_size = 1024
    energy = np.array([
        np.sum(np.abs(data[i:i+frame_size]))
        for i in range(0, len(data), frame_size)
    ])

    peaks = np.where(energy > np.mean(energy))[0]

    if len(peaks) > 1:
        intervals = np.diff(peaks)
        avg_interval = np.mean(intervals)
        bpm = 60 * samplerate / (avg_interval * frame_size)
    else:
        bpm = 0

    bpm = round(float(min(max(bpm, 60), 200)), 1)

    # Énergie globale
    energy_score = np.clip(np.mean(np.abs(data)) * 10, 0, 1)

    # FFT simple
    fft = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1/samplerate)

    # Brillance = hautes fréquences
    brightness = np.clip(np.sum(fft[freqs > 5000]) / np.sum(fft), 0, 1)

    # Largeur spectrale
    spectral_width = np.clip(np.std(fft) / np.max(fft), 0, 1)

    # Bruit
    noise = np.clip(np.var(data), 0, 1)

    # Retombée fréquentielle
    spectral_decay = np.clip(np.mean(fft[:len(fft)//4]) / np.mean(fft), 0, 1)

    # Tempo normalisé pour radar
    tempo_norm = bpm / 200

    # -----------------------------
    # 🎨 CONSTRUCTION FIGURE
    # -----------------------------

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#2f3542")

    # ===== Bloc texte =====
    fig.text(0.05, 0.75, "ARPOZ Selecta", fontsize=28, weight="bold", color="white")

    info_text = (
        f"Tempo : {bpm} BPM\n"
        f"Énergie : {round(energy_score,2)}\n"
        f"Brillance : {round(brightness,2)}\n"
        f"Sensation : dynamique"
    )

    fig.text(0.05, 0.55, info_text, fontsize=14, color="cyan")

    # ===== Radar =====
    categories = [
        "Tempo",
        "Brillance",
        "Largeur spectrale",
        "Bruit",
        "Énergie",
        "Retombée freq."
    ]

    values = [
        tempo_norm,
        brightness,
        spectral_width,
        noise,
        energy_score,
        spectral_decay
    ]

    values += values[:1]

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax_radar = plt.subplot(2,2,2, polar=True)
    ax_radar.set_facecolor("#2f3542")
    ax_radar.plot(angles, values, linewidth=2)
    ax_radar.fill(angles, values, alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_yticklabels([])

    # ===== Waveform =====
    ax_wave = plt.subplot(2,1,2)
    ax_wave.plot(data[:3000], color="cyan")
    ax_wave.set_title("Waveform", color="white")
    ax_wave.set_facecolor("#1e272e")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
