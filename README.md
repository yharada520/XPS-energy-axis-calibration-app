# XPS Energy Axis Calibration (ISO 15472-like)

**English edition (v5)** of a small Flask web app for energy-axis calibration of XPS using Au/Ag/Cu reference spectra. It reads your CSV files (multi-spectra format), estimates peak centers with a simple pseudo-Voigt model, aggregates ISO-like metrics (Δ1_Au, Δ2_Ag, Δ4_Cu), and visualizes:

- Zoomable peak overlays (Plotly) for Au/Ag/Cu
- Control chart of Δ vs. timestamp (user-adjustable limits)
- Linearity residuals (ref − mean) bar chart
- SQLite-backed run history (instrument/operator/notes)

> **Note**: Keep your internal v3 private if any. This repository targets v5 (public, MIT). v1/v2/v4 may also be published if they contain no confidential data.

---

## 1 Features

- **CSV inspector**: Detects candidate signal columns (S1..Sk) from each CSV.
- **Peak estimation**: Fits a lightweight pseudo-Voigt around reference energy within a configurable window.
- **ISO-like metrics**: Δ1_Au, Δ2_Ag, Δ4_Cu; linearity error proxy ε₂; repeatability proxy σ_R; per-element U95.
- **Control chart**: Instrument/date-range filters + adjustable Δ limit / Warn limit from the UI.
- **Linearity residuals**: Vertical axis defined as [(reference − mean)][eV].
- **Persistence**: SQLite for runs and per-spectrum triplets.

---

## 2 Quick start

```bash
# (Windows PowerShell)
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app_xps_calibration_flask_v5_en.py
# open http://127.0.0.1:5000
````

Mac/Linux:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app_xps_calibration_flask_v5_en.py
```

---

## 3 CSV format

- Flexible header is allowed. The app autodetects the first numeric row.
- After that, columns must be: `Energy, S1, S2, ... Sk`.
- Units: Energy in eV; intensities as counts (arbitrary units).

Examples are recommended under `examples/` (synthetic preferred; no internal data).

---

## 4 Usage flow

1. **Upload** Au/Ag/Cu CSVs on */*.
2. Click **Inspect columns** to populate S1..Sk.
3. Optionally select columns; set **reference BEs**, **peak window**, **FWHM/η bounds**, and **N/random seed**.
4. Click **Analyze & Generate**.
5. On */analyze* result page:

   - Review **Summary** and **Details**.
   - Explore **zoomable peak plots** for each element.
   - Adjust **instrument/date/limits** and click **Update** to refresh the **control chart**; **Download PNG** if needed.
   - Check **Linearity residuals** chart.
   - See **History** for accumulated runs.

---

## 5) Outputs

- `data/charts/control_chart.png` (and variants via filters)
- `data/charts/linearity_residuals.png`
- `data/db/xps_calibration.db` (SQLite)

> These are ignored by git via `.gitignore`.

---

## 6) Development notes

- The peak model is a compact pseudo-Voigt mixture; for production use, tune constraints and add robust error handling if spectra are noisy.
- Plotly is embedded using CDN for convenience.
- For fonts (Matplotlib), the app tries `japanize-matplotlib` when available; otherwise defaults to Matplotlib.

---

## 7) Security & privacy

- Uploaded files are not persisted by default (`SAVE_UPLOADS=False`). If you enable it, ensure files contain **no sensitive data**. Keep internal v3 non-public.

---

## 8) License

MIT License. See `LICENSE`.

## 9) Citation (optional)

Add a `CITATION.cff` if you plan to archive on Zenodo.
