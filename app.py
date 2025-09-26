import os
import io
import json
import base64
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Docker-safe
import matplotlib.pyplot as plt

import requests
from datetime import timedelta
from flask import Flask, render_template, request, send_file
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# ================== Azure ML ==================
ENDPOINT = os.getenv("AML_ENDPOINT", "")
API_KEY  = os.getenv("AML_KEY", "")

SCORE_MIN = float(os.getenv("ANOM_SCORE_MIN", "0.8"))
ATYPICAL_MODE = os.getenv("ATYPICAL_MODE", "intersect").lower()  # intersect|series|model
FACTOR_IQR = float(os.getenv("FACTOR_IQR", "1.5"))

# ======= Top anomalías (ranking + NMS temporal) =======
HIGHLIGHT_TOP_K = int(os.getenv("HIGHLIGHT_TOP_K", "30"))
HIGHLIGHT_TOP_PCT = float(os.getenv("HIGHLIGHT_TOP_PCT", "0"))
HIGHLIGHT_MIN_SEP_HRS = int(os.getenv("HIGHLIGHT_MIN_SEP_HRS", "12"))
HIGHLIGHT_W_SCORE = float(os.getenv("HIGHLIGHT_W_SCORE", "1.0"))
HIGHLIGHT_W_EXCESS = float(os.getenv("HIGHLIGHT_W_EXCESS", "1.0"))

# ======= Regla por valor (TU REGLA: >100 o ≈ 0) =======
VALUE_RULE_HIGH = float(os.getenv("VALUE_RULE_HIGH", "100"))
VALUE_RULE_MARK_ZERO = os.getenv("VALUE_RULE_MARK_ZERO", "1").lower() in ("1","true","yes","y")
VALUE_RULE_ZERO_TOL = float(os.getenv("VALUE_RULE_ZERO_TOL", "1e-9"))
# 'only' (solo regla), 'and' (intersección con IQR/modelo), 'or' (unión)
VALUE_RULE_MODE = os.getenv("VALUE_RULE_MODE", "only").lower()

EXPECTED_COLS = [
    "Hourly_Date",
    "VolUnCorrected",
    "Pressure",
    "Temperature",
    "MaxPressure",
    "MinPressure",
    "MaxFlow",
    "MinFlow",
    "MaxTemp",
    "MinTemp",
    "VolCorrected_lag1",
    "VolCorrected_lag2",
    "VolCorrected_lag24",
    "VolCorrected_rollmean3",
    "VolCorrected_rollmean6",
    "VolCorrected_rollmean24",
]

SITE_CANDIDATES = ["site","Site","sitio","Sitio","station","Station","station_id",
                   "Estacion","estacion","ID","id","id_sitio"]

# Estado exportación
LAST_ANOMALIES = None
LAST_TITLE_SUFFIX = ""


# ================== IO ==================
def _read_any(file_storage):
    name = file_storage.filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_storage)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_storage)
    if name.endswith(".data"):
        try:
            file_storage.seek(0)
            return pd.read_csv(file_storage)
        except Exception:
            file_storage.seek(0)
            return pd.read_csv(file_storage, delim_whitespace=True, header=None)
    return pd.read_csv(file_storage)


def _filter_by_site(df: pd.DataFrame, site_label: str) -> pd.DataFrame:
    if not site_label:
        return df
    target = str(site_label).strip()
    for c in SITE_CANDIDATES:
        if c in df.columns:
            s = df[c].astype(str)
            mask = (s == target) | (s.str.contains(target, na=False))
            if mask.any():
                return df[mask].copy()
    return df


# ================== Crear lags/rollings si faltan (FIX) ==================
def _ensure_required_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea lags y medias móviles que el endpoint necesita si no existen en el archivo.
    Usa VolCorrected si existe; si no, VolUnCorrected. Ordena por Hourly_Date.
    """
    if "Hourly_Date" not in df.columns:
        raise ValueError("La columna 'Hourly_Date' es obligatoria en el archivo de entrada.")

    base_col = "VolCorrected" if "VolCorrected" in df.columns else (
        "VolUnCorrected" if "VolUnCorrected" in df.columns else None
    )
    if base_col is None:
        raise ValueError("Debe existir 'VolCorrected' o 'VolUnCorrected' en el archivo.")

    work = df.copy()
    work["Hourly_Date"] = pd.to_datetime(work["Hourly_Date"], errors="coerce")
    work = work.sort_values("Hourly_Date")

    y = pd.to_numeric(work[base_col], errors="coerce")

    # Lags requeridos
    if "VolCorrected_lag1" not in work.columns:
        work["VolCorrected_lag1"] = y.shift(1)
    if "VolCorrected_lag2" not in work.columns:
        work["VolCorrected_lag2"] = y.shift(2)
    if "VolCorrected_lag24" not in work.columns:
        work["VolCorrected_lag24"] = y.shift(24)

    # Rolling means requeridos
    if "VolCorrected_rollmean3" not in work.columns:
        work["VolCorrected_rollmean3"] = y.rolling(3, min_periods=3).mean()
    if "VolCorrected_rollmean6" not in work.columns:
        work["VolCorrected_rollmean6"] = y.rolling(6, min_periods=6).mean()
    if "VolCorrected_rollmean24" not in work.columns:
        work["VolCorrected_rollmean24"] = y.rolling(24, min_periods=24).mean()

    # Rellena otras columnas esperadas si faltan (como placeholders)
    for col in EXPECTED_COLS:
        if col not in work.columns:
            if col == "Hourly_Date":
                continue
            work[col] = pd.NA

    return work


# ================== Preparación p/ endpoint ==================
def _prepare_rows(df: pd.DataFrame):
    work = df.copy()
    for col in EXPECTED_COLS:
        if col not in work.columns:
            work[col] = pd.NA
    work = work[EXPECTED_COLS]

    work["Hourly_Date"] = pd.to_datetime(work["Hourly_Date"], errors="coerce")
    work = work.sort_values("Hourly_Date")
    sorted_index = work.index.to_list()

    work["Hourly_Date"] = work["Hourly_Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    num_cols = [c for c in EXPECTED_COLS if c != "Hourly_Date"]
    for c in num_cols:
        work[c] = work[c].astype(str).str.replace(",", ".", regex=False)
        work[c] = pd.to_numeric(work[c], errors="coerce")

    valid_mask = work[num_cols].notna().all(axis=1)
    work_valid = work[valid_mask].copy()
    idx_valid = [sorted_index[i] for i, ok in enumerate(valid_mask.tolist()) if ok]

    rows = work_valid.to_dict(orient="records")
    return rows, idx_valid


def call_azure_ml(rows: list[dict]):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {"rows": rows}
    resp = requests.post(ENDPOINT, headers=headers, data=json.dumps(payload), timeout=90)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "error" in data:
        raise ValueError(f"Endpoint error: {data['error']}")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            pass

    result = None
    if isinstance(data, dict) and "result" in data:
        result = data["result"]
    elif isinstance(data, list):
        result = data
    else:
        for key in ("data", "outputs", "predictions"):
            if isinstance(data, dict) and key in data:
                result = data[key]
                break
    if not isinstance(result, list):
        raise ValueError(f"Invalid response format: {data}")

    labels, scores = [], []
    for item in result:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid result item: {item}")
        a = item.get("anomaly")
        s = item.get("anomaly_score")
        if a is None or s is None:
            raise ValueError(f"Invalid result item (missing keys): {item}")
        labels.append(int(a))
        scores.append(float(s))
    return labels, scores


# ================== Selección atípicos ==================
def _iqr_thresholds(y: pd.Series):
    y = pd.to_numeric(y, errors="coerce")
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + FACTOR_IQR * iqr
    lower = q1 - FACTOR_IQR * iqr
    return lower, upper


def _select_outliers_both_tails(df: pd.DataFrame, series_col: str, is_anom_col: str = "is_anomaly"):
    s = pd.to_numeric(df[series_col], errors="coerce")
    lower_thr, upper_thr = _iqr_thresholds(s)
    m_series = (s < lower_thr) | (s > upper_thr)

    m_model = (df[is_anom_col] == 1)
    if "anomaly_score" in df.columns and SCORE_MIN > 0:
        m_model &= (pd.to_numeric(df["anomaly_score"], errors="coerce") >= SCORE_MIN)

    if ATYPICAL_MODE == "series":
        mask = m_series
    elif ATYPICAL_MODE == "model":
        mask = m_model
    else:  # intersect
        mask = m_series & m_model

    return df[mask].copy(), lower_thr, upper_thr


def _local_extrema_mask(y: pd.Series, win: int = 5) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    w = max(3, win if win % 2 == 1 else win + 1)
    roll_max = y.rolling(w, center=True).max()
    roll_min = y.rolling(w, center=True).min()
    return ((y == roll_max) | (y == roll_min)) & y.notna()


def _select_atipicas_extrema(df: pd.DataFrame, series_col: str, is_anom_col: str = "is_anomaly"):
    out, lo, up = _select_outliers_both_tails(df, series_col, is_anom_col)
    if out.empty:
        return out, lo, up
    mask_ext = _local_extrema_mask(out[series_col], win=5)
    return out[mask_ext].copy(), lo, up


# ================== Regla por VALOR (>100 o ≈ 0) ==================
def _value_rule_mask(df: pd.DataFrame, series_col: str) -> pd.Series:
    y = pd.to_numeric(df[series_col], errors="coerce")
    mask = y > VALUE_RULE_HIGH
    if VALUE_RULE_MARK_ZERO:
        mask = mask | (y.abs() <= VALUE_RULE_ZERO_TOL)
    return mask


# ================== Ranking & NMS ==================
def _robust_norm(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    p1, p99 = np.nanpercentile(s, 1), np.nanpercentile(s, 99)
    rng = max(p99 - p1, 1e-9)
    return (s - p1) / rng


def _nms_time(df: pd.DataFrame, time_col: str, score_col: str, min_sep_hours: int) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col, score_col]).sort_values(score_col, ascending=False)
    kept, last_times = [], []
    sep = pd.Timedelta(hours=min_sep_hours)
    for _, row in d.iterrows():
        t = row[time_col]
        if not last_times or all(abs(t - lt) > sep for lt in last_times):
            kept.append(row)
            last_times.append(t)
    return pd.DataFrame(kept)


def _highlight_top(df_anoms: pd.DataFrame, series_col: str, lower_thr: float, upper_thr: float) -> pd.DataFrame:
    if df_anoms.empty:
        return df_anoms
    d = df_anoms.copy()
    y = pd.to_numeric(d[series_col], errors="coerce")
    excess_high = np.clip(y - upper_thr, 0, None)
    excess_low  = np.clip(lower_thr - y, 0, None)
    excess = np.maximum(excess_high, excess_low)
    ex_n = _robust_norm(excess)
    if "anomaly_score" in d.columns:
        sc_n = _robust_norm(d["anomaly_score"])
    else:
        sc_n = pd.Series(0.0, index=d.index)
    d["__saliency"] = HIGHLIGHT_W_EXCESS * ex_n + HIGHLIGHT_W_SCORE * sc_n
    d = d.sort_values("__saliency", ascending=False)
    if HIGHLIGHT_TOP_PCT > 0:
        k = max(1, int(len(d) * (HIGHLIGHT_TOP_PCT / 100.0)))
        d = d.head(k)
    else:
        d = d.head(max(1, HIGHLIGHT_TOP_K if HIGHLIGHT_TOP_K > 0 else 30))
    d = _nms_time(d, time_col="Hourly_Date", score_col="__saliency", min_sep_hours=HIGHLIGHT_MIN_SEP_HRS)
    return d


# ================== Gráficos ==================
def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _scatter_anoms(ax, df_pts: pd.DataFrame, series_col: str, label: str):
    pts = df_pts.copy()
    pts["__x"] = pd.to_datetime(pts["Hourly_Date"], errors="coerce")
    pts["__y"] = pd.to_numeric(pts[series_col], errors="coerce")
    pts = pts.dropna(subset=["__x", "__y"])
    if not pts.empty:
        ax.scatter(pts["__x"], pts["__y"], marker="x", s=160, color="red",
                   linewidths=2.8, zorder=10, label=label)
    return len(pts)


def _build_plots_from_anoms(df: pd.DataFrame, series_col: str, title_suffix: str,
                            an: pd.DataFrame, lower_thr: float, upper_thr: float):
    x = pd.to_datetime(df.get("Hourly_Date", pd.Series(range(len(df)))), errors="coerce")
    y = pd.to_numeric(df[series_col], errors="coerce").fillna(0.0)

    # Serie completa
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(x, y, linewidth=1, label=f"{series_col} {title_suffix}")
    if not an.empty:
        _scatter_anoms(ax1, an, series_col, f"Anomalías {title_suffix}")
    ax1.axhline(upper_thr, linestyle="--", linewidth=0.9, color="#888", alpha=0.6)
    ax1.axhline(lower_thr, linestyle="--", linewidth=0.9, color="#888", alpha=0.6)
    ax1.set_title(f"Detección de anomalías - {series_col} {title_suffix}")
    ax1.set_xlabel("Fecha"); ax1.set_ylabel(series_col); ax1.legend()
    img1 = _fig_to_base64(fig1)

    # Zoom ±15 días (centrado en anomalías)
    if not an.empty:
        centers = pd.to_datetime(an["Hourly_Date"], errors="coerce").dropna().sort_values()
        center = centers.iloc[len(centers)//2] if not centers.empty else x.dropna().max()
    else:
        center = x.dropna().max()
    if pd.isna(center):
        start, end = None, None
    else:
        start, end = center - timedelta(days=15), center + timedelta(days=15)
    mask = (x >= start) & (x <= end) if start is not None else pd.Series([True]*len(x))

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(x[mask], y[mask], linewidth=1, label=f"{series_col} (Zoom) {title_suffix}")
    anz = an.copy()
    if not anz.empty and start is not None:
        xm = (pd.to_datetime(anz["Hourly_Date"], errors="coerce") >= start) & \
             (pd.to_datetime(anz["Hourly_Date"], errors="coerce") <= end)
        anz = anz[xm]
    if not anz.empty:
        _scatter_anoms(ax2, anz, series_col, f"Anomalías (Zoom) {title_suffix}")
    ax2.axhline(upper_thr, linestyle="--", linewidth=0.9, color="#888", alpha=0.6)
    ax2.axhline(lower_thr, linestyle="--", linewidth=0.9, color="#888", alpha=0.6)
    ax2.set_title(f"Detección de anomalías - {series_col} (Zoom) {title_suffix}")
    ax2.set_xlabel("Fecha"); ax2.set_ylabel(series_col); ax2.legend()
    img2 = _fig_to_base64(fig2)

    return img1, img2


# ================== Cálculo con REGLA DE VALOR ==================
def compute_anomalies(df: pd.DataFrame, series_col: str, is_anom_col: str):
    # Lógica (IQR+modelo+extremos)
    an_raw, lo_thr, up_thr = _select_atipicas_extrema(df, series_col, is_anom_col)
    # Regla valor
    mask_rule = _value_rule_mask(df, series_col)
    an_rule = df[mask_rule].copy()
    # Combinar
    if VALUE_RULE_MODE == "only":
        an_comb = an_rule
    elif VALUE_RULE_MODE == "and":
        an_comb = pd.merge(an_raw, an_rule, how="inner")
    else:
        an_comb = pd.concat([an_raw, an_rule], axis=0).drop_duplicates()
    # Ranking + NMS
    an_final = _highlight_top(an_comb, series_col, lo_thr, up_thr)
    return an_final, lo_thr, up_thr


# ================== Rutas ==================
@app.route("/", methods=["GET", "POST"])
def index():
    global LAST_ANOMALIES, LAST_TITLE_SUFFIX

    if request.method == "POST":
        file = request.files.get("file")
        sitio = request.form.get("site_label", "").strip()
        LAST_TITLE_SUFFIX = f"(Sitio {sitio})" if sitio else ""

        if not file or file.filename == "":
            return render_template("index.html", error="Sube un archivo válido (.csv, .xlsx, .data).")

        # 1) leer y filtrar por sitio (sin fechas)
        raw_df = _read_any(file)
        df_site = _filter_by_site(raw_df, sitio)

        if df_site.empty:
            return render_template("index.html", error="No hay datos para el sitio seleccionado.")

        # 2) generar lags/rollings si no existen y luego preparar filas
        try:
            df_site = _ensure_required_features(df_site)
        except Exception as e:
            return render_template("index.html", error=str(e))

        rows, idx_valid = _prepare_rows(df_site)
        if len(rows) == 0:
            return render_template(
                "index.html",
                error="No hay filas válidas. Asegúrate de tener al menos ~25 filas y que 'Hourly_Date' y la serie (VolCorrected o VolUnCorrected) sean válidas."
            )

        # 3) inferencia
        labels, scores = call_azure_ml(rows)

        # 4) re-alinear
        df = df_site.copy()
        df["is_anomaly"] = 0
        df["anomaly_score"] = np.nan
        df.loc[idx_valid, "is_anomaly"] = [1 if int(x) == 1 else 0 for x in labels]
        df.loc[idx_valid, "anomaly_score"] = scores

        # 5) serie a graficar
        series_col = "VolCorrected" if "VolCorrected" in df.columns else "VolUnCorrected"
        if series_col not in df.columns:
            df[series_col] = df["anomaly_score"]
        if "Hourly_Date" in df.columns:
            df["Hourly_Date"] = pd.to_datetime(df["Hourly_Date"], errors="coerce")
            df = df.sort_values("Hourly_Date")

        # 6) calcular anomalías (con regla valor)
        an_final, lo_thr, up_thr = compute_anomalies(df, series_col, "is_anomaly")
        LAST_ANOMALIES = an_final.copy()

        # 7) imágenes
        img_full, img_zoom = _build_plots_from_anoms(df, series_col, LAST_TITLE_SUFFIX,
                                                     an_final, lo_thr, up_thr)
        sample = df.head(50)
        count_anoms = int(LAST_ANOMALIES.shape[0])

        return render_template("index.html",
                               error=None,
                               img_full=img_full,
                               img_zoom=img_zoom,
                               rows_preview=sample.to_html(index=False, classes="tbl"),
                               anomalies_count=count_anoms)

    # GET
    return render_template("index.html", error=None)


@app.route("/download/xlsx")
def download_xlsx():
    global LAST_ANOMALIES
    if LAST_ANOMALIES is None or LAST_ANOMALIES.empty:
        df = pd.DataFrame(columns=["Mensaje"], data=[["No hay anomalías para exportar"]])
    else:
        df = LAST_ANOMALIES.copy()

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="anomalies")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="anomalies.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/download/pdf")
def download_pdf():
    global LAST_ANOMALIES, LAST_TITLE_SUFFIX
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    story.append(Paragraph(f"Anomalías detectadas {LAST_TITLE_SUFFIX}", styles["Title"]))
    story.append(Spacer(1, 12))

    if LAST_ANOMALIES is None or LAST_ANOMALIES.empty:
        story.append(Paragraph("No hay anomalías para exportar.", styles["Normal"]))
    else:
        df = LAST_ANOMALIES.copy()
        max_rows = 200
        if len(df) > max_rows:
            df = df.head(max_rows)
            story.append(Paragraph(f"Mostrando primeras {max_rows} filas.", styles["Italic"]))
            story.append(Spacer(1, 6))
        data = [df.columns.tolist()] + df.astype(str).values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("ALIGN", (0,0), (-1,0), "CENTER"),
        ]))
        story.append(table)

    doc.build(story)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="anomalies.pdf",
                     mimetype="application/pdf")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
