import matplotlib
matplotlib.use("Agg")

import io
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import soundfile as sf

app = FastAPI()

BRAND_CYAN = "#289dcc"
BG_COLOR = "#2f3542"


@app.get("/")
def root():
    return {"status": "alive"}


@app.post("/analyze/", response_class=Response)
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()
    data, samplerate = sf.read(io.BytesIO(contents))

    # ---------- Nettoyage nom ----------
    filename = os.path.splitext(file.filename)[0]
    filename = re.sub(r"\s\(\d+\)$", "", filename)

    # ---------- Segment 60s → 75s ----------
    segment_duration = 15
    start_second = 60

    start_sample = samplerate * start_second
    end_sample = start_sample + samplerate * segment_duration

    if len(data) > end_sample:
        data = data[start_sample:end_sample]
    else:
        data = data[:samplerate * segment_duration]

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    if len(data) < 2048:
        return Response(content=b"Audio too short", media_type="text/plain")

    # ---------- ANALYSE ----------

    rms = float(np.sqrt(np.mean(data**2)))
    energy_score = np.clip(rms * 10, 0, 1)

    fft = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1/samplerate)

    fft_sum = np.sum(fft) if np.sum(fft) != 0 else 1

    brightness = np.clip(np.sum(fft[freqs > 5000]) / fft_sum, 0, 1)
    spectral_width = np.clip(np.std(fft) / (np.max(fft) if np.max(fft)!=0 else 1), 0, 1)
    noise = np.clip(np.var(data), 0, 1)
    spectral_decay = np.clip(np.mean(fft[:len(fft)//4]) / fft_sum, 0, 1)

    tempo_norm = np.clip(np.mean(np.abs(np.diff(data))) * 20, 0, 1)

    # ---------- FIGURE 16:9 ----------

    fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)

    # ----- LOGO -----
    try:
        logo = mpimg.imread("logo.png")
        ax_logo = fig.add_axes([0.05, 0.80, 0.25, 0.15])
        ax_logo.imshow(logo)
        ax_logo.axis("off")
    except:
        pass

    # ----- TITRE -----
    fig.text(
        0.05, 0.74,
        filename.upper(),
        fontsize=22,
        color="white",
        weight="bold"
    )

    # ----- BARRE BLEUE -----
    ax_bar = fig.add_axes([0.045, 0.58, 0.006, 0.20])
    ax_bar.set_facecolor(BRAND_CYAN)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # ----- TEXTE QUALITATIF -----

    def describe_energy(score):
        if score < 0.2: return "très douce"
        elif score < 0.4: return "plutôt posée"
        elif score < 0.6: return "modérément énergique"
        elif score < 0.8: return "énergique"
        else: return "très énergique"

    def describe_valence(score):
        if score < 0.2: return "très sombre"
        elif score < 0.4: return "plutôt sombre"
        elif score < 0.6: return "neutre"
        elif score < 0.8: return "plutôt lumineuse"
        else: return "très lumineuse"

    profil_text = (
        f"Énergie : {describe_energy(energy_score)}\n"
        f"Brillance : {describe_valence(brightness)}\n"
        f"Sensation : {describe_energy(tempo_norm)}"
    )

    fig.text(
        0.06, 0.72,
        profil_text,
        fontsize=15,
        color="white",
        va="top",
        linespacing=1.6
    )

    # ----- RADAR PROPRE -----

    categories = [
        "Tempo",
        "Brillance",
        "Largeur spectrale",
        "Bruit",
        "Énergie",
        "Retombée freq."
    ]

    features = np.array([
        tempo_norm,
        brightness,
        spectral_width,
        noise,
        energy_score,
        spectral_decay
    ])

    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
    radar_angles = np.concatenate((angles, [angles[0]]))
    radar_values = np.concatenate((features, [features[0]]))

    ax_radar = fig.add_axes([0.55, 0.42, 0.35, 0.45], polar=True)
    ax_radar.set_facecolor(BG_COLOR)

    ax_radar.plot(radar_angles, radar_values, color="white", linewidth=2)
    ax_radar.fill(radar_angles, radar_values, color=BRAND_CYAN, alpha=0.45)

    ax_radar.set_ylim(0, 1)

    # Supprimer degrés et valeurs
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels([])
    ax_radar.set_yticklabels([])

    ax_radar.grid(color="white", alpha=0.3)
    ax_radar.spines["polar"].set_color("white")

    # Labels éloignés
    for angle, label in zip(angles, categories):
        ax_radar.text(
            angle,
            1.25,
            label,
            color="white",
            fontsize=12,
            ha="center",
            va="center"
        )

    # ----- WAVEFORM -----

    ax_wave = fig.add_axes([0.05, 0.08, 0.90, 0.25])
    ax_wave.set_facecolor("#1e272e")

    times = np.linspace(0, len(data)/samplerate, num=len(data))
    step = max(len(data)//3000, 1)

    ax_wave.fill_between(times[::step], data[::step], color=BRAND_CYAN)

    ax_wave.set_xlim(0, len(data)/samplerate)
    ax_wave.set_ylim(-1, 1)

    ax_wave.set_title("Waveform", color="white")
    ax_wave.tick_params(colors="white")

    for spine in ax_wave.spines.values():
        spine.set_color("white")

    # ----- EXPORT -----

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
