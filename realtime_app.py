"""
SAFE-AD: Real-Time Spacecraft Anomaly Detection
Live Sensor Stream Simulation Dashboard
Author: K R Janani | MSc Data Science 
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import stft
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time, ast, warnings, io
warnings.filterwarnings("ignore")
np.random.seed(42); torch.manual_seed(42)

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAFE-AD | Live Monitor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #F8FAFC; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #1B3A6B 0%, #1A7F74 100%);
    padding: 1.2rem 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.main-header h1 {
    color: white !important;
    font-size: 1.8rem !important;
    font-weight: 700;
    margin: 0;
}
.main-header p {
    color: #A5C8C4;
    margin: 0;
    font-size: 0.9rem;
}

/* Status badge */
.status-normal {
    background: #DCFCE7;
    color: #15803D;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
    display: inline-block;
    border: 2px solid #22C55E;
}
.status-anomaly {
    background: #FEE2E2;
    color: #B91C1C;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
    display: inline-block;
    border: 2px solid #EF4444;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0%   { opacity: 1.0; }
    50%  { opacity: 0.6; }
    100% { opacity: 1.0; }
}

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric-val {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
}
.metric-lbl {
    font-size: 0.72rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
}

/* Alert box */
.alert-box {
    background: #FEF2F2;
    border: 2px solid #EF4444;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.alert-box h3 { color: #B91C1C; margin: 0 0 0.5rem 0; }
.alert-box p  { color: #7F1D1D; margin: 0.2rem 0; font-size: 0.9rem; }

/* Normal box */
.normal-box {
    background: #F0FDF4;
    border: 2px solid #22C55E;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
}
.normal-box p { color: #14532D; margin: 0; font-size: 0.95rem; }

/* Info box */
.info-box {
    background: #EFF6FF;
    border-left: 4px solid #1B3A6B;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    color: #1E3A5F;
}

/* Sensor badge */
.sensor-badge {
    background: #1A7F74;
    color: white;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin: 2px;
}

/* Severity */
.sev-high   { background:#FEE2E2; color:#B91C1C; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.85rem; }
.sev-medium { background:#FEF3C7; color:#92400E; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.85rem; }
.sev-low    { background:#DCFCE7; color:#15803D; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device("cpu")
WS = 256; STEP = 64  # smaller step for smoother real-time feel

# ── Helper Functions ───────────────────────────────────────────────
def create_windows(data, ws=WS, step=STEP):
    return np.array([data[i:i+ws]
                     for i in range(0, len(data)-ws+1, step)])

def minmax(x):
    mn = x.min(); mx = x.max()
    return (x - mn) / (mx - mn + 1e-8)

def compute_stft_window(win, nperseg=64):
    F = win.shape[1]; et=lt=ht=0
    se = np.zeros(F)
    for f in range(F):
        _, _, Z = stft(win[:, f], nperseg=nperseg)
        sp = np.abs(Z); mid = sp.shape[0]//2
        e=sp.mean(); l=sp[:mid].mean(); h=sp[mid:].mean()
        se[f]=e; et+=e; lt+=l; ht+=h
    return et/F, lt/F, ht/F, se

def anom_type(lf, hf):
    if   hf > 1.3*lf: return "HF — Spike / Noise",    "#EF4444"
    elif lf > 1.3*hf: return "LF — Drift / Shift",    "#2563EB"
    else:              return "Mixed — Complex Pattern","#8E44AD"

def severity_label(sc, p33, p66):
    if   sc > p66: return "HIGH",   "sev-high"
    elif sc > p33: return "MEDIUM", "sev-medium"
    else:          return "LOW",    "sev-low"

def fig2bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110,
                bbox_inches="tight", facecolor="white")
    buf.seek(0); plt.close(fig)
    return buf

# ── Transformer ────────────────────────────────────────────────────
class TSD(Dataset):
    def __init__(self, w):
        self.w = torch.tensor(w, dtype=torch.float32)
    def __len__(self): return len(self.w)
    def __getitem__(self, i): return self.w[i]

def mask_inp(x, r=0.25):
    xm = x.clone()
    m = (torch.rand(x.shape[0], x.shape[1], 1).expand_as(x) < r)
    xm[m] = 0.0; return xm, m

class MT(nn.Module):
    def __init__(self, nf, d=64, h=4, nl=2):
        super().__init__()
        self.inp = nn.Linear(nf, d)
        self.pos = nn.Embedding(512, d)
        enc = nn.TransformerEncoderLayer(d, h, d*4, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, nl)
        self.out = nn.Linear(d, nf)
    def forward(self, x):
        B, T, _ = x.shape
        p = torch.arange(T).unsqueeze(0).expand(B, T)
        return self.out(self.enc(self.inp(x) + self.pos(p)))

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ SAFE-AD Control Panel")
    st.markdown("---")

    st.markdown("### 📂 Data Upload")
    train_file  = st.file_uploader("Training data (.npy)", type=["npy"])
    test_file   = st.file_uploader("Test data (.npy)",     type=["npy"])
    labels_file = st.file_uploader("Labels CSV (optional)",type=["csv"])

    st.markdown("---")
    st.markdown("### ⚙️ Stream Settings")
    channel_name = st.text_input("Channel ID", value="M-6")
    stream_speed = st.select_slider(
        "Stream Speed",
        options=["Very Slow", "Slow", "Normal", "Fast", "Very Fast"],
        value="Normal"
    )
    speed_map = {
        "Very Slow": 0.5, "Slow": 0.25,
        "Normal": 0.08, "Fast": 0.03, "Very Fast": 0.01
    }
    DELAY = speed_map[stream_speed]

    epochs = st.slider("Training epochs", 5, 15, 8)

    st.markdown("---")
    train_btn  = st.button("🔧  Train SAFE-AD", use_container_width=True)
    stream_btn = st.button("▶️  Start Live Stream", use_container_width=True)
    stop_btn   = st.button("⏹️  Stop Stream",       use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **SAFE-AD**
    K R Janani · 126150019
    MSc Data Science · SRC
    *Base: Pattern Recognition 2024*
    """)

# ── Header ─────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🛰️ SAFE-AD — Live Spacecraft Anomaly Monitor</h1>
    <p>STFT-Aware Fusion Ensemble · Real-Time Sensor Stream Simulation · NASA SMAP/MSL</p>
</div>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────
for key, val in {
    "trained": False, "model": None, "iso": None,
    "mu": None, "sd": None, "streaming": False,
    "stream_done": False, "alerts": [],
    "iso_tr_mean": 0, "iso_tr_std": 1,
    "tf_tr_mean":  0, "tf_tr_std":  1,
    "st_tr_mean":  0, "st_tr_std":  1,
    "p33": 0.3, "p66": 0.6,
    "y_true": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── TRAIN ──────────────────────────────────────────────────────────
if train_btn:
    if train_file is None:
        st.error("Please upload training data first!")
        st.stop()

    train_raw = np.load(train_file)
    if train_raw.ndim == 1: train_raw = train_raw.reshape(-1, 1)
    NF = train_raw.shape[1]

    mu = train_raw.mean(0); sd = train_raw.std(0) + 1e-8
    trn = (train_raw - mu) / sd
    trw = create_windows(trn)

    prog = st.progress(0, "🔧 Training SAFE-AD...")

    # STFT baseline
    prog.progress(5, "📡 Computing STFT baseline...")
    st_scores = []
    for w in trw:
        e,_,_,_ = compute_stft_window(w)
        st_scores.append(e)
    st_scores = np.array(st_scores)
    st.session_state.st_tr_mean = st_scores.mean()
    st.session_state.st_tr_std  = st_scores.std() + 1e-8

    # Isolation Forest
    prog.progress(20, "🌲 Training Isolation Forest...")
    feats = []
    for w in trw:
        fv = []
        for f in range(NF):
            s = w[:, f]
            fv += [s.mean(), s.std(), np.abs(np.fft.rfft(s)).mean()]
        feats.append(fv)
    feats = np.array(feats)
    iso = IsolationForest(n_estimators=100, contamination=0.05,
                          random_state=42, n_jobs=-1)
    iso.fit(feats)
    iso_tr = -iso.decision_function(feats)
    st.session_state.iso_tr_mean = iso_tr.mean()
    st.session_state.iso_tr_std  = iso_tr.std() + 1e-8

    # Transformer
    model = MT(nf=NF, d=64, h=4, nl=2).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf_fn = nn.MSELoss()
    ld    = DataLoader(TSD(trw), batch_size=32, shuffle=True)
    model.train()
    for ep in range(epochs):
        for b in ld:
            b = b.to(DEVICE); xm, mk = mask_inp(b)
            pr = model(xm)
            loss = lf_fn(pr[mk], b[mk])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        prog.progress(20 + int((ep+1)/epochs*65),
                      f"🤖 Training Transformer epoch {ep+1}/{epochs}...")

    model.eval()
    tf_errs = []
    with torch.no_grad():
        for b in DataLoader(TSD(trw), batch_size=32):
            b = b.to(DEVICE); xm, _ = mask_inp(b)
            pr = model(xm)
            tf_errs.extend(((pr-b)**2).mean(dim=(1,2)).cpu().numpy())
    tf_errs = np.array(tf_errs)
    st.session_state.tf_tr_mean = tf_errs.mean()
    st.session_state.tf_tr_std  = tf_errs.std() + 1e-8

    st.session_state.model   = model
    st.session_state.iso     = iso
    st.session_state.mu      = mu
    st.session_state.sd      = sd
    st.session_state.trained = True
    st.session_state.alerts  = []

    prog.progress(100, "✅ SAFE-AD trained and ready!")
    time.sleep(0.5); prog.empty()
    st.success("✅ SAFE-AD is trained! Click **▶️ Start Live Stream** to begin monitoring.")

# ── IDLE STATE ─────────────────────────────────────────────────────
if not st.session_state.trained and not stream_btn:
    c1, c2, c3 = st.columns(3)
    for col, step, desc in [
        (c1, "Step 1: Upload + Train",
         "Upload .npy train & test files → Click 🔧 Train SAFE-AD"),
        (c2, "Step 2: Start Stream",
         "Click ▶️ Start Live Stream to begin real-time simulation"),
        (c3, "Step 3: Watch Live",
         "See anomaly score rise in real-time with ALERTS when detected"),
    ]:
        col.markdown(f"""
        <div style='background:white;border:1px solid #E2E8F0;border-radius:10px;
                    padding:1rem;text-align:center;'>
            <b style='color:#1B3A6B;'>{step}</b><br>
            <span style='color:#64748B;font-size:0.85rem;'>{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    📌 <b>How it works:</b> Upload your NASA SMAP/MSL .npy channel files using the sidebar.
    Train the model, then start the live stream to see SAFE-AD detect anomalies in real-time,
    just like a spacecraft engineer watching a mission control dashboard.
    </div>""", unsafe_allow_html=True)

# ── LIVE STREAM ────────────────────────────────────────────────────
if stream_btn:
    if not st.session_state.trained:
        st.error("⚠️ Please train SAFE-AD first using the sidebar!")
        st.stop()
    if test_file is None:
        st.error("⚠️ Please upload test data!")
        st.stop()

    test_raw = np.load(test_file)
    if test_raw.ndim == 1: test_raw = test_raw.reshape(-1, 1)
    NF   = test_raw.shape[1]
    NPTS = len(test_raw)

    # Labels
    y_true = None
    if labels_file is not None:
        try:
            ldf = pd.read_csv(labels_file)
            row = ldf[ldf["chan_id"] == channel_name]
            if not row.empty:
                y_true = np.zeros(NPTS, dtype=int)
                for s, e in ast.literal_eval(
                        row.iloc[0]["anomaly_sequences"]):
                    y_true[s:e] = 1
        except: pass
    st.session_state.y_true = y_true

    mu = st.session_state.mu
    sd = st.session_state.sd
    ten = (test_raw - mu) / sd
    windows = create_windows(ten)
    N_WINS = len(windows)

    model = st.session_state.model
    iso   = st.session_state.iso

    # Pre-compute all scores
    with st.spinner("Computing anomaly scores..."):
        all_st  = []
        all_lf  = []
        all_hf  = []
        all_se  = []
        all_iso = []
        all_tf  = []

        for w in windows:
            e,lf,hf,se = compute_stft_window(w)
            all_st.append(e); all_lf.append(lf)
            all_hf.append(hf); all_se.append(se)

        all_st = np.array(all_st)
        all_lf = np.array(all_lf)
        all_hf = np.array(all_hf)
        all_se = np.array(all_se)

        feats = []
        for w in windows:
            fv = []
            for f in range(NF):
                s = w[:, f]
                fv += [s.mean(), s.std(), np.abs(np.fft.rfft(s)).mean()]
            feats.append(fv)
        feats = np.array(feats)
        all_iso = -iso.decision_function(feats)

        model.eval()
        with torch.no_grad():
            for b in DataLoader(TSD(windows), batch_size=32):
                b = b.to(DEVICE); xm,_ = mask_inp(b)
                pr = model(xm)
                all_tf.extend(((pr-b)**2).mean(dim=(1,2)).cpu().numpy())
        all_tf = np.array(all_tf)

        # Normalise
        st_n  = (all_st  - st.session_state.st_tr_mean)  / st.session_state.st_tr_std
        iso_n = (all_iso - st.session_state.iso_tr_mean) / st.session_state.iso_tr_std
        tf_n  = (all_tf  - st.session_state.tf_tr_mean)  / st.session_state.tf_tr_std
        ens   = 0.45*minmax(tf_n) + 0.30*minmax(iso_n) + 0.25*minmax(st_n)

        # Threshold
        THR    = np.percentile(ens, 88)
        p33    = np.percentile(ens[ens > THR], 33) if (ens>THR).sum()>0 else 0.4
        p66    = np.percentile(ens[ens > THR], 66) if (ens>THR).sum()>0 else 0.7
        st.session_state.p33 = p33
        st.session_state.p66 = p66

    # ── Dashboard layout ──────────────────────────────────────────
    # Top row — status + key metrics
    status_ph = st.empty()
    m1,m2,m3,m4,m5 = st.columns(5)
    w_ph   = m1.empty()
    sc_ph  = m2.empty()
    at_ph  = m3.empty()
    sv_ph  = m4.empty()
    al_ph  = m5.empty()

    # Charts row
    ch1, ch2 = st.columns([3, 1])
    chart_ph   = ch1.empty()
    sensor_ph  = ch2.empty()

    # Alert log
    st.markdown("### 🚨 Alert Log")
    log_ph = st.empty()

    # ── Stream loop ───────────────────────────────────────────────
    alerts       = []
    scores_so_far = []
    times_so_far  = []
    gt_so_far     = []

    for wi in range(N_WINS):
        # Check stop
        if stop_btn:
            break

        score    = float(ens[wi])
        lf_v     = float(all_lf[wi])
        hf_v     = float(all_hf[wi])
        se_v     = all_se[wi]
        is_anom  = score > THR
        t_start  = wi * STEP
        t_end    = min(t_start + WS, NPTS)
        t_mid    = (t_start + t_end) // 2

        gt_label = int(y_true[t_mid]) if y_true is not None else -1

        scores_so_far.append(score)
        times_so_far.append(t_mid)
        gt_so_far.append(gt_label)

        at_label, at_color = anom_type(lf_v, hf_v)
        sv_label, sv_class = severity_label(score, p33, p66)
        top3 = list(se_v.argsort()[::-1][:3])

        # Status badge
        if is_anom:
            status_ph.markdown(f"""
            <div style='text-align:center; padding:0.6rem;
                        background:#FEF2F2; border-radius:10px;
                        border:2px solid #EF4444; margin-bottom:0.5rem;'>
                <span style='font-size:1.5rem;'>🚨</span>
                <span style='font-size:1.2rem; font-weight:700;
                             color:#B91C1C; margin-left:8px;'>
                    ANOMALY DETECTED — Channel {channel_name}
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            status_ph.markdown(f"""
            <div style='text-align:center; padding:0.6rem;
                        background:#F0FDF4; border-radius:10px;
                        border:2px solid #22C55E; margin-bottom:0.5rem;'>
                <span style='font-size:1.5rem;'>✅</span>
                <span style='font-size:1.2rem; font-weight:700;
                             color:#15803D; margin-left:8px;'>
                    NORMAL OPERATION — Channel {channel_name}
                </span>
            </div>""", unsafe_allow_html=True)

        # Metrics
        pct = int(100 * wi / N_WINS)
        w_ph.markdown(f"""<div class='metric-card'>
            <p class='metric-val' style='color:#1B3A6B;'>{pct}%</p>
            <p class='metric-lbl'>Stream Progress</p></div>""",
            unsafe_allow_html=True)

        sc_color = "#EF4444" if is_anom else "#1A7F74"
        sc_ph.markdown(f"""<div class='metric-card'>
            <p class='metric-val' style='color:{sc_color};'>{score:.3f}</p>
            <p class='metric-lbl'>Anomaly Score</p></div>""",
            unsafe_allow_html=True)

        at_ph.markdown(f"""<div class='metric-card'>
            <p class='metric-val' style='font-size:0.9rem;color:{at_color};'>
                {at_label}</p>
            <p class='metric-lbl'>Anomaly Type</p></div>""",
            unsafe_allow_html=True)

        sv_ph.markdown(f"""<div class='metric-card'>
            <p class='metric-val' style='font-size:1.1rem;'><span class='{sv_class}'>{sv_label}</span></p>
            <p class='metric-lbl'>Severity</p></div>""",
            unsafe_allow_html=True)

        n_alerts = len(alerts)
        al_ph.markdown(f"""<div class='metric-card'>
            <p class='metric-val' style='color:#{"EF4444" if n_alerts>0 else "1A7F74"};'>
                {n_alerts}</p>
            <p class='metric-lbl'>Total Alerts</p></div>""",
            unsafe_allow_html=True)

        # Live score chart
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.fill_between(times_so_far, scores_so_far,
                        alpha=0.15, color="#1A7F74")
        ax.plot(times_so_far, scores_so_far,
                color="#1A7F74", linewidth=1.5)
        ax.axhline(THR, color="#EF4444", linestyle="--",
                   linewidth=1.5, label=f"Threshold ({THR:.3f})")

        # Ground truth shading
        if y_true is not None:
            in_seg = False
            for ti, gt in zip(times_so_far, gt_so_far):
                if gt == 1 and not in_seg:
                    seg_s = ti; in_seg = True
                elif gt == 0 and in_seg:
                    ax.axvspan(seg_s, ti, color="#EF4444",
                               alpha=0.15)
                    in_seg = False
            if in_seg:
                ax.axvspan(seg_s, times_so_far[-1],
                           color="#EF4444", alpha=0.15)

        # Highlight anomaly windows
        anom_t = [t for t, s in zip(times_so_far, scores_so_far)
                  if s > THR]
        if anom_t:
            ax.scatter(anom_t,
                       [scores_so_far[times_so_far.index(t)]
                        for t in anom_t],
                       color="#EF4444", s=25, zorder=5)

        ax.set_xlim(0, NPTS)
        ax.set_ylim(0, max(1.0, max(scores_so_far)*1.1))
        ax.set_title(f"Live Anomaly Score — Channel {channel_name}  "
                     f"(Window {wi+1}/{N_WINS})",
                     fontweight="bold", fontsize=11)
        ax.set_xlabel("Time Index"); ax.set_ylabel("Score")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        chart_ph.image(fig2bytes(fig), use_container_width=True)

        # Sensor attribution
        fig2, ax2 = plt.subplots(figsize=(3.5, 3.2))
        colors_s = ["#1A7F74" if i in top3 else "#CBD5E1"
                    for i in range(min(NF, 25))]
        ax2.bar(range(min(NF, 25)), se_v[:25],
                color=colors_s, alpha=0.9)
        ax2.set_title("Sensor Attribution\n(Teal = Top 3)",
                      fontweight="bold", fontsize=10)
        ax2.set_xlabel("Sensor ID", fontsize=9)
        ax2.set_ylabel("Energy", fontsize=9)
        ax2.grid(axis="y", alpha=0.3)
        ax2.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        sensor_ph.image(fig2bytes(fig2), use_container_width=True)

        # Alert log
        if is_anom:
            alerts.append({
                "Window": wi+1,
                "Time":   t_mid,
                "Score":  round(score, 4),
                "Type":   at_label,
                "Severity": sv_label,
                "Top Sensors": str(top3),
            })

        if alerts:
            adf = pd.DataFrame(alerts[::-1])  # newest first
            log_ph.dataframe(adf, use_container_width=True,
                             hide_index=True)
        else:
            log_ph.markdown("""<div class='normal-box'>
            <p>✅ No anomalies detected yet — system operating normally</p>
            </div>""", unsafe_allow_html=True)

        time.sleep(DELAY)

    # ── Stream complete ────────────────────────────────────────────
    st.success(f"✅ Stream complete! Processed {N_WINS} windows.")

    # Final metrics
    if y_true is not None and len(scores_so_far) > 0:
        st.markdown("### 📊 Final Evaluation Metrics")

        # Build time-series score
        ts = np.zeros(NPTS)
        tc = np.zeros(NPTS)
        for wi2, sc in enumerate(scores_so_far):
            s = wi2 * STEP; e = min(s+WS, NPTS)
            ts[s:e] += sc; tc[s:e] += 1
        ts = ts / (tc + 1e-8)

        pred = (ts > THR).astype(int)
        # Point adjustment
        def gseg(y):
            segs=[]; fl=False
            for i,v in enumerate(y):
                if v==1 and not fl: s2=i; fl=True
                elif v==0 and fl: segs.append((s2,i)); fl=False
            if fl: segs.append((s2,len(y)))
            return segs
        pred_pa = pred.copy()
        for s2,e2 in gseg(y_true):
            if np.any(pred[s2:e2]==1): pred_pa[s2:e2]=1

        f1  = f1_score(y_true, pred_pa, zero_division=0)
        try:   roc = roc_auc_score(y_true, ts)
        except: roc = float("nan")
        try:   pr  = average_precision_score(y_true, ts)
        except: pr  = float("nan")

        mc1,mc2,mc3,mc4 = st.columns(4)
        for col, val, lbl, color in [
            (mc1, f"{f1:.4f}",  "F1 Score (PA)",  "#1A7F74"),
            (mc2, f"{roc:.4f}", "ROC-AUC",        "#2563EB"),
            (mc3, f"{pr:.4f}",  "PR-AUC",         "#7C3AED"),
            (mc4, str(len(alerts)), "Total Alerts", "#EF4444"),
        ]:
            col.markdown(f"""<div class='metric-card'>
                <p class='metric-val' style='color:{color};'>{val}</p>
                <p class='metric-lbl'>{lbl}</p></div>""",
                unsafe_allow_html=True)

    if alerts:
        st.markdown("### 📋 Complete Anomaly Report")
        adf = pd.DataFrame(alerts)
        st.dataframe(adf, use_container_width=True, hide_index=True)
        csv = adf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Anomaly Report CSV",
            data=csv,
            file_name=f"SAFE_AD_alerts_{channel_name}.csv",
            mime="text/csv"
        )

# ── Footer ─────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#94A3B8; font-size:0.78rem;'>
<b style='color:#1A7F74;'>SAFE-AD</b> &nbsp;·&nbsp;
STFT-Aware Fusion Ensemble for Anomaly Detection &nbsp;·&nbsp;
K R Janani · 126150019 · MSc Data Science · SRC &nbsp;·&nbsp;
Base Paper: Pattern Recognition, Elsevier 2024
</div>""", unsafe_allow_html=True)
