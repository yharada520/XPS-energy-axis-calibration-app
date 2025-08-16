# app_xps_calibration_flask_v5_en.py
# ------------------------------------------------------------
# ISO 15472 XPS Energy Axis Calibration (Au/Ag/Cu)
# Flask + SQLite + Matplotlib + Plotly (zoomable peak plots)
#
# v5 highlights (EN edition)
#  - Control chart: instrument & date range filters + user-adjustable
#    delta limit and warning limit (from UI)
#  - HTML templates split under templates/*.html
#  - Linearity residuals: vertical axis = (reference − mean) [eV]
# ------------------------------------------------------------
import io, os, csv, math, random, sqlite3
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from flask import Flask, request, render_template, jsonify, send_file, url_for
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plotly.graph_objs as go
from plotly.offline import plot as plotly_offline_plot

# Optional Japanese font support (harmless if not installed)
try:
    import japanize_matplotlib  # noqa: F401
except Exception:
    pass

# --------------------- Paths & constants ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(DATA_DIR, "db")
CHART_DIR = os.path.join(DATA_DIR, "charts")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

for d in [DATA_DIR, DB_DIR, CHART_DIR, UPLOAD_DIR]:
    os.makedirs(d, exist_ok=True)

DB_PATH = os.path.join(DB_DIR, "xps_calibration.db")
DEFAULT_CHART_PATH = os.path.join(CHART_DIR, "control_chart.png")
LINEARITY_PATH = os.path.join(CHART_DIR, "linearity_residuals.png")

DEFAULT_DELTA_LIMIT = 0.20
DEFAULT_WARN_LIMIT = 0.078

SAVE_UPLOADS = False  # set True if you want to keep uploaded CSVs

# --------------------- Flask ---------------------
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB

# --------------------- Utils ---------------------
def fmt4(v: Optional[float]) -> str:
    return "nan" if (v is None or (isinstance(v, float) and (not np.isfinite(v)))) else f"{v:.4f}"

# ------------- Peak model (pseudo-Voigt like) -------------
def voigt_like(x, amp, center, fwhm, eta):
    # very simple pseudo-Voigt mixture for peak finding
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    lorentz = 1.0 / (1.0 + ((x - center) / (0.5 * fwhm)) ** 2)
    return amp * (eta * lorentz + (1.0 - eta) * gauss)

# quick-and-robust single-peak estimation around reference e0
# returns (center, height)
def fit_peak_center_and_height(energy: np.ndarray,
                               intensity: np.ndarray,
                               peak_window_eV: float,
                               fwhm_bounds: Tuple[float,float],
                               eta_bounds: Tuple[float,float],
                               e0: Optional[float]=None) -> Tuple[float, float]:
    if e0 is None:
        idx0 = int(np.argmax(intensity))
        e0 = float(energy[idx0])
    idx0 = int(np.argmin(np.abs(energy - e0)))
    mask = (energy > e0 - peak_window_eV) & (energy < e0 + peak_window_eV)
    x = energy[mask]; y = intensity[mask]
    if len(x) < 8:
        return e0, float(y.max() if len(y) else intensity[idx0])

    amp0 = float(y.max())
    fwhm0 = np.clip((peak_window_eV / 2), fwhm_bounds[0], fwhm_bounds[1])
    eta0 = 0.5
    p0 = [amp0, e0, fwhm0, eta0]
    bounds = (
        [0.0, e0 - peak_window_eV, fwhm_bounds[0], eta_bounds[0]],
        [np.inf, e0 + peak_window_eV, fwhm_bounds[1], eta_bounds[1]],
    )
    try:
        popt, _ = curve_fit(voigt_like, x, y, p0=p0, bounds=bounds, maxfev=20000)
        amp, cen, fwhm, eta = popt
        yfit = voigt_like(x, *popt)
        return float(cen), float(yfit.max())
    except Exception:
        return e0, amp0

# ============== CSV reader (multi-spectra) ==================
# Flexible header, then numeric lines: Energy,S1,S2,...,Sk
# Returns DataFrame with columns [Energy, S1..]
def read_multi_spectra_csv(file_bytes: bytes) -> pd.DataFrame:
    text = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    lines = list(reader)

    # find the first numeric row (Energy must be float)
    start_idx = None
    for i, row in enumerate(lines):
        if not row:
            continue
        try:
            _ = float(row[0])
            start_idx = i
            break
        except Exception:
            continue
    if start_idx is None:
        raise ValueError("Failed to detect numeric rows. Ensure first column is Energy (float).")

    df = pd.DataFrame(lines[start_idx:])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')

    cols = ["Energy"] + [f"S{i}" for i in range(1, df.shape[1])]
    df.columns = cols
    return df

# ================= ISO-like metrics =========================
def compute_deltas(mean_positions: Dict[str, float], refs: Dict[str, float]) -> Dict[str, float]:
    return {
        "Δ1_Au": mean_positions.get("Au", np.nan) - refs["Au"],
        "Δ2_Ag": mean_positions.get("Ag", np.nan) - refs["Ag"],
        "Δ4_Cu": mean_positions.get("Cu", np.nan) - refs["Cu"],
    }

def compute_epsilon2(deltas: Dict[str, float]) -> float:
    # simple aggregated linearity error proxy
    vals = [abs(deltas.get(k, np.nan)) for k in ("Δ1_Au","Δ2_Ag","Δ4_Cu")]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float('nan')

# repeatability proxy (std of mean positions)
def compute_sigma_R(per_elem_centers: Dict[str, List[float]]) -> float:
    all_means = []
    for k in ("Au","Ag","Cu"):
        arr = per_elem_centers.get(k, [])
        if len(arr) > 0:
            all_means.append(float(np.mean(arr)))
    if len(all_means) <= 1:
        return float('nan')
    return float(np.std(all_means, ddof=1))

# approximate U95 per element (2*std / sqrt(n))
def compute_u95(per_elem_centers: Dict[str, List[float]]) -> Dict[str, float]:
    res = {}
    for k in ("Au","Ag","Cu"):
        arr = per_elem_centers.get(k, [])
        if len(arr) <= 1:
            res[k] = float('nan')
        else:
            res[k] = 2.0 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr))
    return res

# ================== SQLite storage ==========================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT,
  instrument TEXT,
  operator TEXT,
  seed INTEGER,
  ref_au REAL,
  ref_ag REAL,
  ref_cu REAL,
  mean_Au REAL,
  mean_Ag REAL,
  mean_Cu REAL,
  d1_Au REAL,
  d2_Ag REAL,
  d4_Cu REAL,
  epsilon2 REAL,
  sigma_R REAL,
  u95_Au REAL,
  u95_Ag REAL,
  u95_Cu REAL,
  residual_Au REAL,
  residual_Ag REAL,
  residual_Cu REAL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS triplets (
  run_id INTEGER,
  element TEXT,
  name TEXT,
  center REAL,
  height REAL
);
"""

def init_db(db_path: str = DB_PATH):
    with sqlite3.connect(db_path) as con:
        con.executescript(SCHEMA_SQL)


def insert_run(ts: str,
               instrument: str,
               operator: str,
               seed: Optional[int],
               refs: Dict[str,float],
               stats: Dict[str,float],
               per_elem_triplets: Dict[str, List[Tuple[str,float,float]]],
               u95s: Dict[str,float],
               notes: str,
               db_path: str = DB_PATH) -> int:
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO runs (
              ts, instrument, operator, seed,
              ref_au, ref_ag, ref_cu,
              mean_Au, mean_Ag, mean_Cu,
              d1_Au, d2_Ag, d4_Cu,
              epsilon2, sigma_R,
              u95_Au, u95_Ag, u95_Cu,
              residual_Au, residual_Ag, residual_Cu,
              notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ts, instrument, operator, seed,
                refs['Au'], refs['Ag'], refs['Cu'],
                stats.get('mean_Au'), stats.get('mean_Ag'), stats.get('mean_Cu'),
                stats.get('Δ1_Au'), stats.get('Δ2_Ag'), stats.get('Δ4_Cu'),
                stats.get('epsilon2'), stats.get('sigma_R'),
                u95s.get('Au'), u95s.get('Ag'), u95s.get('Cu'),
                stats.get('resid_Au'), stats.get('resid_Ag'), stats.get('resid_Cu'),
                notes,
            )
        )
        run_id = cur.lastrowid
        rows = []
        for elem in ("Au","Ag","Cu"):
            for (name, cen, h) in per_elem_triplets.get(elem, []):
                rows.append((run_id, elem, name, float(cen), float(h)))
        if rows:
            cur.executemany("INSERT INTO triplets (run_id, element, name, center, height) VALUES (?,?,?,?,?)", rows)
        con.commit()
        return int(run_id)


def fetch_runs(db_path: str = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query("SELECT * FROM runs ORDER BY id DESC", con)
    return df


def fetch_instruments(db_path: str = DB_PATH) -> List[str]:
    try:
        df = fetch_runs(db_path)
        vals = sorted([v for v in df['instrument'].dropna().unique().tolist() if str(v).strip() != ""])
        return vals
    except Exception:
        return []


def fetch_deltas_series_filtered(instrument: Optional[str] = None,
                                 start: Optional[str] = None,
                                 end: Optional[str] = None,
                                 db_path: str = DB_PATH) -> pd.DataFrame:
    df = fetch_runs(db_path)
    if instrument:
        df = df[df['instrument'] == instrument]
    if start:
        df = df[df['ts'] >= start]
    if end:
        df = df[df['ts'] <= end]
    df = df[['ts','d1_Au','d2_Ag','d4_Cu']].copy()
    return df

# =================== Plots =====================
def make_control_chart(df: pd.DataFrame, delta_limit: float, warn_limit: float, out_path: str) -> Optional[str]:
    if df is None or df.empty:
        return None
    ts = pd.to_datetime(df['ts'])
    fig, ax = plt.subplots(figsize=(8,4), dpi=150)
    ax.plot(ts, df['d1_Au'], marker='o', linestyle='-', label='Δ1_Au (Au)')
    ax.plot(ts, df['d2_Ag'], marker='o', linestyle='-', label='Δ2_Ag (Ag)')
    ax.plot(ts, df['d4_Cu'], marker='o', linestyle='-', label='Δ4_Cu (Cu)')
    ax.axhline(0.0, linestyle='--', linewidth=1)
    for y, ls, lab in [(delta_limit,'--','Δ limit'),(-delta_limit,'--','-Δ limit'),(warn_limit,':','Warn'),(-warn_limit,':','-Warn')]:
        ax.axhline(y, linestyle=ls, linewidth=1, label=None)
    ax.set_ylabel('Δ (eV)')
    ax.set_xlabel('Timestamp')
    ax.set_title('Control Chart of Δ (ISO-like)')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def make_linearity_plot(residuals: Dict[str, float], out_path: str) -> Optional[str]:
    if residuals is None:
        return None
    fig, ax = plt.subplots(figsize=(4,3), dpi=150)
    xs = ["Au","Ag","Cu"]
    ys = [residuals.get("Au", np.nan), residuals.get("Ag", np.nan), residuals.get("Cu", np.nan)]
    ax.bar(xs, ys)
    ax.axhline(0.0, linestyle='--', linewidth=1)
    ax.set_ylabel('Residual (ref − mean) [eV]')
    ax.set_title('Linearity Residuals')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def build_plotly_peak_figure(energy: np.ndarray,
                             df: pd.DataFrame,
                             use_cols: List[str],
                             centers: List[float],
                             heights: List[float],
                             element: str) -> Optional[str]:
    if df is None or len(use_cols) == 0:
        return None
    fig = go.Figure()
    for c in use_cols:
        fig.add_trace(go.Scatter(x=energy, y=df[c].to_numpy(), mode="lines", name=c, line=dict(width=1)))
    fig.add_trace(go.Scatter(x=centers, y=heights, mode="markers+text",
                             text=[f"{element}:{i+1}" for i in range(len(centers))], textposition="top center",
                             name=f"Estimated centers ({element})"))
    fig.update_layout(title=f"{element} spectra & estimated peak centers",
                      xaxis_title="Binding Energy (eV)", yaxis_title="Counts")
    fig.update_xaxes(autorange="reversed")  # XPS style
    return plotly_offline_plot(fig, include_plotlyjs='cdn', output_type='div')

# =================== Routes ================================
@app.route("/", methods=["GET"])
def index():
    init_db(DB_PATH)
    return render_template("index.html")

@app.route("/inspect", methods=["POST"])
def inspect():
    """Return candidate signal columns (S1..Sk) per element from three CSVs."""
    result = {}
    for elem, key in (("Au","au_file"), ("Ag","ag_file"), ("Cu","cu_file")):
        f = request.files.get(key)
        if not f:
            result[elem] = []
            continue
        raw = f.read()
        if SAVE_UPLOADS:
            fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{elem}.csv"
            with open(os.path.join(UPLOAD_DIR, fname), "wb") as wf:
                wf.write(raw)
        df = read_multi_spectra_csv(raw)
        cols = [c for c in df.columns if c != "Energy"]
        result[elem] = cols
    return jsonify(result)

@app.route("/analyze", methods=["POST"])
def analyze():
    init_db(DB_PATH)

    # files
    au_file = request.files.get("au_file")
    ag_file = request.files.get("ag_file")
    cu_file = request.files.get("cu_file")
    if not (au_file and ag_file and cu_file):
        return "CSV for Au/Ag/Cu are required.", 400

    # params
    n_select = int(request.form.get("n_select", 7))
    randomize = True if request.form.get("randomize") == "on" else False
    seed = request.form.get("seed", "").strip()
    seed_val = int(seed) if seed != "" else None

    ref_au = float(request.form.get("ref_au", 83.95))
    ref_ag = float(request.form.get("ref_ag", 368.22))
    ref_cu = float(request.form.get("ref_cu", 932.62))
    peak_window_eV = float(request.form.get("peak_window_eV", 1.5))
    fwhm_min = float(request.form.get("fwhm_min", 0.3))
    fwhm_max = float(request.form.get("fwhm_max", 2.5))
    eta_min = float(request.form.get("eta_min", 0.0))
    eta_max = float(request.form.get("eta_max", 1.0))

    instrument = request.form.get("instrument", "")
    operator = request.form.get("operator", "")
    notes = request.form.get("notes", "")

    # selected columns
    au_cols = request.form.getlist("au_cols")
    ag_cols = request.form.getlist("ag_cols")
    cu_cols = request.form.getlist("cu_cols")

    # read CSVs
    au_raw = au_file.read(); ag_raw = ag_file.read(); cu_raw = cu_file.read()
    if SAVE_UPLOADS:
        ts_save = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(UPLOAD_DIR, f"{ts_save}_Au.csv"), "wb") as wf: wf.write(au_raw)
        with open(os.path.join(UPLOAD_DIR, f"{ts_save}_Ag.csv"), "wb") as wf: wf.write(ag_raw)
        with open(os.path.join(UPLOAD_DIR, f"{ts_save}_Cu.csv"), "wb") as wf: wf.write(cu_raw)

    au_df = read_multi_spectra_csv(au_raw)
    ag_df = read_multi_spectra_csv(ag_raw)
    cu_df = read_multi_spectra_csv(cu_raw)

    fwhm_bounds = (fwhm_min, fwhm_max)
    eta_bounds = (eta_min, eta_max)
    refs = {"Au":ref_au, "Ag":ref_ag, "Cu":ref_cu}

    def analyze_one(df, label, selected):
        energy = df["Energy"].to_numpy()
        all_cols = [c for c in df.columns if c != "Energy"]
        # determine columns to use
        use_cols = selected if selected else all_cols[:]
        if seed_val is not None:
            random.seed(seed_val)
        if randomize and len(use_cols) > n_select:
            use_cols = random.sample(use_cols, n_select)
        names, centers, heights = [], [], []
        for c in use_cols:
            cen, h = fit_peak_center_and_height(
                energy, df[c].to_numpy(),
                peak_window_eV=peak_window_eV,
                fwhm_bounds=fwhm_bounds, eta_bounds=eta_bounds
            )
            names.append(f"{label}:{c}")
            centers.append(cen)
            heights.append(h)
        return energy, use_cols, names, centers, heights

    au_energy, au_use, au_names, au_centers, au_heights = analyze_one(au_df, os.path.basename(au_file.filename), au_cols)
    ag_energy, ag_use, ag_names, ag_centers, ag_heights = analyze_one(ag_df, os.path.basename(ag_file.filename), ag_cols)
    cu_energy, cu_use, cu_names, cu_centers, cu_heights = analyze_one(cu_df, os.path.basename(cu_file.filename), cu_cols)

    mean_positions = {
        "Au": float(np.mean(au_centers)) if au_centers else float('nan'),
        "Ag": float(np.mean(ag_centers)) if ag_centers else float('nan'),
        "Cu": float(np.mean(cu_centers)) if cu_centers else float('nan'),
    }
    deltas = compute_deltas(mean_positions, refs)
    epsilon2 = compute_epsilon2(deltas)
    sigma_R = compute_sigma_R({"Au":au_centers, "Ag":ag_centers, "Cu":cu_centers})
    u95s = compute_u95({"Au":au_centers, "Ag":ag_centers, "Cu":cu_centers})

    residuals = {
        "Au": refs["Au"] - mean_positions["Au"],
        "Ag": refs["Ag"] - mean_positions["Ag"],
        "Cu": refs["Cu"] - mean_positions["Cu"],
    }

    # save linearity residuals bar chart
    linearity = make_linearity_plot(residuals, LINEARITY_PATH)
    linearity_available = linearity is not None and os.path.exists(LINEARITY_PATH)

    # summary (plain text block)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    stats = {
        'mean_Au': mean_positions['Au'], 'mean_Ag': mean_positions['Ag'], 'mean_Cu': mean_positions['Cu'],
        'Δ1_Au': deltas['Δ1_Au'], 'Δ2_Ag': deltas['Δ2_Ag'], 'Δ4_Cu': deltas['Δ4_Cu'],
        'epsilon2': epsilon2, 'sigma_R': sigma_R,
        'resid_Au': residuals['Au'], 'resid_Ag': residuals['Ag'], 'resid_Cu': residuals['Cu'],
    }

    per_elem_triplets = {
        'Au': list(zip(au_names, au_centers, au_heights)),
        'Ag': list(zip(ag_names, ag_centers, ag_heights)),
        'Cu': list(zip(cu_names, cu_centers, cu_heights)),
    }

    run_id = insert_run(
        ts, instrument or "", operator or "", seed_val,
        refs, stats, per_elem_triplets, u95s, notes or "",
        db_path=DB_PATH
    )

    # default control chart (no filter, default limits)
    df_series_all = fetch_deltas_series_filtered()
    chart_path = make_control_chart(df_series_all, DEFAULT_DELTA_LIMIT, DEFAULT_WARN_LIMIT, DEFAULT_CHART_PATH)
    chart_available = chart_path is not None and os.path.exists(chart_path)

    # Plotly peak figures (HTML div)
    peaks_div_au = build_plotly_peak_figure(au_energy, au_df, au_use, au_centers, au_heights, "Au") if au_use else None
    peaks_div_ag = build_plotly_peak_figure(ag_energy, ag_df, ag_use, ag_centers, ag_heights, "Ag") if ag_use else None
    peaks_div_cu = build_plotly_peak_figure(cu_energy, cu_df, cu_use, cu_centers, cu_heights, "Cu") if cu_use else None

    # summary text lines
    summary = []
    summary.append(f"[Run ID] {run_id}  /  Timestamp: {ts}")
    summary.append(f"Instrument: {instrument or '-'}  /  Operator: {operator or '-'}  /  Seed: {seed_val}")
    summary.append(f"Reference BEs (eV): Au={refs['Au']}, Ag={refs['Ag']}, Cu={refs['Cu']}")
    summary.append("")
    summary.append("# of adopted spectra  Au: {}  Ag: {}  Cu: {}".format(len(au_use), len(ag_use), len(cu_use)))
    summary.append("")
    summary.append("Mean peak position (eV):  Au={}  Ag={}  Cu={}".format(
        fmt4(mean_positions["Au"]), fmt4(mean_positions["Ag"]), fmt4(mean_positions["Cu"]) ))
    summary.append("Δ (eV):  Δ1_Au={}  Δ2_Ag={}  Δ4_Cu={}".format(
        fmt4(deltas["Δ1_Au"]), fmt4(deltas["Δ2_Ag"]), fmt4(deltas["Δ4_Cu"]) ))
    summary.append(f"Linearity error ε₂ (eV): {fmt4(epsilon2)}")
    summary.append(f"Repeatability σ_R (eV): {fmt4(sigma_R)}")
    summary.append("U95 (approx, eV):  Au={}  Ag={}  Cu={}".format(
        fmt4(u95s["Au"]), fmt4(u95s["Ag"]), fmt4(u95s["Cu"]) ))
    summary.append("")
    summary.append("Linearity residuals (ref − mean, eV):  Au={}  Ag={}  Cu={}".format(
        fmt4(residuals["Au"]), fmt4(residuals["Ag"]), fmt4(residuals["Cu"]) ))
    summary.append("")
    summary.append(f"Notes: {notes or '-'}")

    instruments = fetch_instruments(DB_PATH)
    chart_url = url_for('chart_png')

    return render_template(
        "result.html",
        summary="\n".join(summary),
        details=[["Au", n, float(c), float(h)] for (n,c,h) in per_elem_triplets['Au']] +
                [["Ag", n, float(c), float(h)] for (n,c,h) in per_elem_triplets['Ag']] +
                [["Cu", n, float(c), float(h)] for (n,c,h) in per_elem_triplets['Cu']],
        chart_available=chart_available,
        chart_url=chart_url,
        instruments=instruments,
        instrument_sel="",
        start_sel="",
        end_sel="",
        delta_limit_init=DEFAULT_DELTA_LIMIT,
        warn_limit_init=DEFAULT_WARN_LIMIT,
        peaks_div_au=peaks_div_au,
        peaks_div_ag=peaks_div_ag,
        peaks_div_cu=peaks_div_cu,
        linearity_available=linearity_available,
        kpi_epsilon2=f"Linearity error ε₂ (eV): {fmt4(epsilon2)}",
    )

# -------- Control chart PNG (with filters & limits) --------
@app.route("/chart.png")
def chart_png():
    instrument = request.args.get("instrument") or None
    start = request.args.get("start") or None
    end = request.args.get("end") or None
    try:
        delta_limit = float(request.args.get("delta", DEFAULT_DELTA_LIMIT))
    except Exception:
        delta_limit = DEFAULT_DELTA_LIMIT
    try:
        warn_limit = float(request.args.get("warn", DEFAULT_WARN_LIMIT))
    except Exception:
        warn_limit = DEFAULT_WARN_LIMIT

    df = fetch_deltas_series_filtered(instrument=instrument, start=start, end=end, db_path=DB_PATH)
    key = f"chart_{instrument or 'ALL'}_{start or 'NA'}_{end or 'NA'}_{delta_limit}_{warn_limit}.png".replace(" ", "_")
    tmp_path = os.path.join(CHART_DIR, key)
    path = make_control_chart(df, delta_limit=delta_limit, warn_limit=warn_limit, out_path=tmp_path)
    if path and os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return "No data to plot.", 404

@app.route("/chart_download")
def chart_download():
    instrument = request.args.get("instrument") or None
    start = request.args.get("start") or None
    end = request.args.get("end") or None
    try:
        delta_limit = float(request.args.get("delta", DEFAULT_DELTA_LIMIT))
    except Exception:
        delta_limit = DEFAULT_DELTA_LIMIT
    try:
        warn_limit = float(request.args.get("warn", DEFAULT_WARN_LIMIT))
    except Exception:
        warn_limit = DEFAULT_WARN_LIMIT

    df = fetch_deltas_series_filtered(instrument=instrument, start=start, end=end, db_path=DB_PATH)
    key = f"chart_{instrument or 'ALL'}_{start or 'NA'}_{end or 'NA'}_{delta_limit}_{warn_limit}.png".replace(" ", "_")
    tmp_path = os.path.join(CHART_DIR, key)
    path = make_control_chart(df, delta_limit=delta_limit, warn_limit=warn_limit, out_path=tmp_path)
    if path and os.path.exists(path):
        return send_file(path, mimetype="image/png", as_attachment=True, download_name=f"{key}")
    return "No data to download.", 404

@app.route("/linearity.png")
def linearity_png():
    if os.path.exists(LINEARITY_PATH):
        return send_file(LINEARITY_PATH, mimetype="image/png")
    return "Linearity plot not found.", 404

@app.route("/history", methods=["GET"])
def history():
    init_db(DB_PATH)
    df = fetch_runs(db_path=DB_PATH)
    rows = df.fillna("").values.tolist() if df is not None and not df.empty else []
    return render_template("history.html", rows=rows)

# ---------------------- main -------------------------------
if __name__ == "__main__":
    init_db(DB_PATH)
    app.run(host="127.0.0.1", port=5000, debug=True)
