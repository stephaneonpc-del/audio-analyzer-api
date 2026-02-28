import matplotlib
matplotlib.use("Agg")

import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

    # 🔥 Limite 15 secondes max
    max_samples = samplerate * 15
    data = data[:max_samples]

    # 🔥 Mono si stéréo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Sécurité si audio trop court
    if len(data) < 2048:
        return Response(content=b"Audio too short", media_type="text/plain")

    # -----------------------------
    # 🎵 ANALYSE SIMPLE ET LÉGÈRE
    # -----------------------------

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

    energy_score = float(np.clip(np.mean(np.abs(data)) * 10, 0, 1))

    fft = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1/samplerate)

    fft_sum = np.sum(fft) if np.sum(fft) != 0 else 1

    brightness = float(np.clip(np.sum(fft[freqs > 5000]) / fft_sum, 0, 1))
    spectral_width = float(np.clip(np.std(fft) / (np.max(fft) if np.max(fft) != 0 else 1), 0, 1))
    noise = float(np.clip(np.var(data), 0, 1))
    spectral_decay = float(np.clip(np.mean(fft[:len(fft)//4]) / fft_sum, 0, 1))

    tempo_norm = bpm / 200

    # -----------------------------
    # 🎨 CONSTRUCTION FIGURE PRO
    # -----------------------------

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor("#2f3542")

    # ===== Logo PNG =====
    try:
        logo = mpimg.imread("logo.png")
        logo = logo[::2, ::2]  # réduit légèrement si trop grand
        fig.figimage(logo, xo=40, yo=650, zorder=10, alpha=1)
    except:
        pass

    # ===== Nom fichier =====
    filename = file.filename if file.filename else "Audio File"

    fig.text(
        0.05, 0.88,
        filename.upper(),
        fontsize=18,
        color="white",
        weight="bold"
    )

    # ===== Barre verticale cyan =====
    fig.patches.extend([
        plt.Rectangle(
            (0.045, 0.60),
            0.006,
            0.20,
            transform=fig.transFigure,
            color="cyan",
            alpha=0.9
        )
    ])

    # ===== Bloc texte structuré =====

    label_x = 0.06
    value_x = 0.16
    y_start = 0.75
    line_gap = 0.05

    labels = ["Tempo :", "Énergie :", "Brillance :", "Sensation :"]
    values_text = [
        f"{bpm} BPM",
        f"{round(energy_score,2)}",
        f"{round(brightness,2)}",
        "dynamique"
    ]

    for i, (lab, val) in enumerate(zip(labels, values_text)):
        y = y_start - i * line_gap
        fig.text(label_x, y, lab, fontsize=15, color="white")
        fig.text(value_x, y, val, fontsize=15, color="cyan")

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
    ax_radar.plot(angles, values, linewidth=2, color="cyan")
    ax_radar.fill(angles, values, alpha=0.25, color="cyan")
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, color="white")
    ax_radar.set_yticklabels([])
    ax_radar.spines["polar"].set_color("white")

    # ===== Waveform =====

    ax_wave = plt.subplot(2,1,2)
    ax_wave.plot(data[:3000], color="cyan", linewidth=1)
    ax_wave.set_title("Waveform", color="white", fontsize=14)
    ax_wave.set_facecolor("#1e272e")
    ax_wave.tick_params(colors="white")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
