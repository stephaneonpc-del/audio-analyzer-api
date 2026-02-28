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
BG_COLOR = "#303440"

@app.get("/")
def root():
    return {"status": "alive"}


# ===============================
# 🎧 DESCRIPTIONS TEXTUELLES
# ===============================

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

def describe_dance(score):
    if score < 0.2: return "très peu dansant"
    elif score < 0.4: return "plutôt introspectif"
    elif score < 0.6: return "modérément dansant"
    elif score < 0.8: return "dansant"
    else: return "très dansant"

def describe_groove(score):
    if score < 0.3: return "très changeant et imprévisible"
    elif score < 0.5: return "changeant et dynamique"
    elif score < 0.7: return "assez stable"
    else: return "très stable et régulier"


@app.post("/analyze/", response_class=Response)
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()
    data, sr = sf.read(io.BytesIO(contents))

    # ---- Nettoyage nom fichier
    filename = os.path.splitext(file.filename)[0]
    filename = re.sub(r"\s\(\d+\)$", "", filename)

    # ---- Limite 15 sec
    data = data[:sr * 15]

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    if len(data) < 2048:
        return Response(content=b"Audio too short", media_type="text/plain")

    # ===============================
    # 🎵 FEATURES OPTIMISÉES
    # ===============================

    # RMS
    rms = float(np.sqrt(np.mean(data**2)))
    energy_score = min(rms * 10, 1)

    # FFT
    fft = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1/sr)
    fft_sum = np.sum(fft) if np.sum(fft) != 0 else 1

    spectral_centroid = np.sum(freqs * fft) / fft_sum
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft) / fft_sum)
    zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(data)))) / 2
    rolloff = freqs[np.where(np.cumsum(fft) >= 0.85 * fft_sum)[0][0]]

    # Fake rhythmic stability (léger)
    onset_env = np.abs(np.diff(data))
    rhythmic_stability = 1 / (1 + np.std(onset_env))

    # Fake danceability
    danceability = min((energy_score * 0.4 + rhythmic_stability * 0.6), 1)

    # Fake valence (basé sur brillance)
    valence_estimate = min((spectral_centroid / 5000), 1)

    # ===============================
    # 📊 RADAR NORMALISÉ
    # ===============================

    features = np.array([
        min((spectral_centroid / 5000), 1),
        min((spectral_bandwidth / 5000), 1),
        min(zero_crossing_rate * 10, 1),
        energy_score,
        min((rolloff / 10000), 1),
        danceability
    ])

    labels = [
        "Brillance",
        "Largeur spectrale",
        "Bruit",
        "Énergie",
        "Retombée fréquentielle",
        "Danceability"
    ]

    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
    radar_features = features.tolist() + [features[0]]
    radar_angles = angles.tolist() + [angles[0]]

    # ===============================
    # 🎨 FIGURE 16:9
    # ===============================

    fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)

    # ---- Logo
    try:
        logo = mpimg.imread("logo.png")
        ax_logo = fig.add_axes([0.05, 0.80, 0.25, 0.15])
        ax_logo.imshow(logo)
        ax_logo.axis("off")
    except:
        pass

    # ---- Titre
    fig.text(
        0.07,
        0.74,
        filename.upper(),
        ha="left",
        va="top",
        color="white",
        fontsize=24,
        fontweight="bold"
    )

    # ---- Profil texte
    profil_text = (
        f"Énergie : {describe_energy(energy_score)}\n"
        f"Groove : {describe_groove(rhythmic_stability)}\n"
        f"Sensation : {describe_dance(danceability)}\n"
        f"Ambiance : {describe_valence(valence_estimate)}"
    )

    fig.text(
        0.07,
        0.65,
        profil_text,
        ha="left",
        va="top",
        color="white",
        fontsize=15,
        linespacing=1.6
    )

    # ---- Barre bleue
    ax_bar = fig.add_axes([0.06, 0.52, 0.004, 0.16])
    ax_bar.set_facecolor(BRAND_CYAN)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # ---- Radar
    ax_radar = fig.add_axes([0.55, 0.45, 0.35, 0.42], polar=True)
    ax_radar.set_facecolor(BG_COLOR)

    ax_radar.fill(radar_angles, radar_features, color=BRAND_CYAN, alpha=0.45)
    ax_radar.plot(radar_angles, radar_features, color="white", linewidth=2)

    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticklabels([])
    ax_radar.xaxis.grid(True, color="white", alpha=0.35)
    ax_radar.yaxis.grid(True, color="white", alpha=0.35)
    ax_radar.spines['polar'].set_color("white")

    for angle, label in zip(angles, labels):
        ax_radar.text(angle, 1.15, label,
                      size=12, color="white",
                      ha="center", va="center")

    # ---- Waveform
    ax_wave = fig.add_axes([0.05, 0.08, 0.90, 0.25])
    ax_wave.set_facecolor("#1e2128")

    times = np.linspace(0, len(data)/sr, num=len(data))
    step = max(len(data)//3000, 1)

    ax_wave.fill_between(times[::step], data[::step], color=BRAND_CYAN, alpha=0.9)

    ax_wave.set_xlim(0, len(data)/sr)
    ax_wave.set_ylim(-1, 1)
    ax_wave.set_title("Waveform", color="white", fontsize=16)
    ax_wave.tick_params(colors='white')

    for spine in ax_wave.spines.values():
        spine.set_color("white")

    # ---- Export
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
