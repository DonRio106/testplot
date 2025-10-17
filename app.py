# -*- coding: utf-8 -*-
"""
Combined Activity Detection (Reaming + Backreaming, single blue color)
Author: Rio Gunawan

Rules:
  Pump ON            : TFLO > TFLO_ON_THRESHOLD (default 10 gpm)
  Excursion (combined, OR):
      (DBTM - CDEPTH) >= +REAM_DELTA_FT          OR
      (CDEPTH - DBTM) >= BACK_DELTA_FT
  Minimum Duration   : MIN_ACTIVITY_SECONDS (default 10 s)

Outputs (written next to the input Excel by default):
  - <base>_segments_blue.csv
      Columns: t_start, t_end, duration_sec, excursion_ft
  - <base>_overview_blue.png
      5-panel overview (TFLO/SPPA/ECD/ROP5/BPOS) with blue strips over BPOS & TFLO
  - <base>_bpos_strip_blue.png
      Compact BPOS strip with blue shading
  - <base>_interactive_blue.html
      Interactive Plotly chart (via CDN), with range slider & selector

Usage:
  python activity_detection_blue.py --excel "time_12.25 inch HS.xlsx"

Optional:
  python activity_detection_blue.py --excel "file.xlsx" --outdir "./results" \
    --tflo 10 --minsec 10 --ream 3 --back 20 --alpha 0.22 --color "#1f77b4"
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates


# ----------------------------
# Defaults (you can override via CLI flags)
# ----------------------------
DEFAULT_TFLO_ON_THRESHOLD     = 10.0   # gpm (Pump ON)
DEFAULT_MIN_ACTIVITY_SECONDS  = 10     # s   (min continuous segment)
DEFAULT_REAM_DELTA_FT         = 3.0    # ft  (DBTM - CDEPTH >= +3 ft)
DEFAULT_BACK_DELTA_FT         = 20.0   # ft  (CDEPTH - DBTM >= 20 ft)
SENTINEL                      = -999.25
STRIP_COLOR_DEFAULT           = "#1f77b4"  # single blue
STRIP_ALPHA_DEFAULT           = 0.22


# ----------------------------
# Data loading & cleaning
# ----------------------------
def load_and_clean_excel(path: Path) -> pd.DataFrame:
    """Load historian Excel, build timestamp, replace sentinels, coerce numerics."""
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    # Build timestamp from Date or Time or first column
    if "Date" in df.columns:
        ts = pd.to_datetime(df["Date"], errors="coerce")
    elif "Time" in df.columns:
        ts = pd.to_datetime(df["Time"], errors="coerce")
    else:
        ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df["timestamp"] = ts

    # Replace sentinel -> NaN and coerce numerics for needed channels
    df = df.replace(SENTINEL, np.nan)
    for c in ["TFLO", "SPPA", "ECD", "ROP5", "DBTM", "BPOS", "TVDE", "CDEPTH"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows without time and sort
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# ----------------------------
# Detection (combined activity)
# ----------------------------
def detect_activity_segments(
    df: pd.DataFrame,
    tflo_thr: float,
    min_sec: int,
    ream_ft: float,
    back_ft: float,
) -> pd.DataFrame:
    """
    Detect contiguous combined activity segments given Pump ON and excursion rules.
    Returns a DataFrame with columns: t_start, t_end, duration_sec, excursion_ft.
    """
    # Required columns
    for col in ["TFLO", "DBTM", "CDEPTH"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    pump_on = (df["TFLO"] > tflo_thr).fillna(False)
    delta   = df["DBTM"] - df["CDEPTH"]
    cond_any = (delta >= ream_ft) | ((-delta) >= back_ft)

    mask = pump_on & cond_any & df["DBTM"].notna() & df["CDEPTH"].notna()

    segments = []
    if mask.any():
        m = mask.astype(int)
        dm = m.diff().fillna(0)

        starts = dm[dm == 1].index.tolist()
        ends   = dm[dm == -1].index.tolist()

        # Edge handling (series starts/ends in True)
        if mask.iloc[0]:
            starts = [mask.index[0]] + starts
        if mask.iloc[-1]:
            ends = ends + [mask.index[-1]]

        for s, e in zip(starts, ends):
            t0 = df.loc[s, "timestamp"]
            t1 = df.loc[e, "timestamp"]
            dur_sec = (t1 - t0).total_seconds()

            if dur_sec >= min_sec:
                # |delta| (absolute excursion vs CDEPTH) max inside segment
                dsub = (df["DBTM"] - df["CDEPTH"]).iloc[s:e+1].values
                excursion_ft = float(np.nanmax(np.maximum(dsub, -dsub)))
                segments.append({
                    "t_start": t0,
                    "t_end"  : t1,
                    "duration_sec": dur_sec,
                    "excursion_ft": excursion_ft,
                })

    seg_df = pd.DataFrame(segments).sort_values("t_start").reset_index(drop=True)
    return seg_df


# ----------------------------
# Plots — Matplotlib
# ----------------------------
def plot_overview_with_blue_strips(df: pd.DataFrame, seg_df: pd.DataFrame,
                                   out_png: Path, color: str, alpha: float) -> Path:
    """
    5-panel overview (TFLO, SPPA, ECD, ROP5, BPOS) with blue activity strips overlay
    on BPOS and TFLO panels. X-axis shows date+time.
    """
    plt.style.use("seaborn-v0_8")
    fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    def panel(ax, col, line_color, label):
        if col in df.columns and df[col].notna().any():
            ax.plot(df["timestamp"], df[col], color=line_color, lw=1)
            ax.set_ylabel(label)
            ax.grid(True, which="both", alpha=0.25)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    panel(axs[0], "TFLO", "teal",   "TFLO (gpm)")
    panel(axs[1], "SPPA", "crimson","SPPA (psi)")
    panel(axs[2], "ECD",  "purple", "ECD (ppg)")
    panel(axs[3], "ROP5", "navy",   "ROP5 (ft/h)")
    panel(axs[4], "BPOS", "black",  "BPOS (ft)")
    axs[-1].set_xlabel("Date & Time")

    # Blue shading on BPOS + TFLO
    if not seg_df.empty:
        ymax = float(df["BPOS"].max(skipna=True)) if "BPOS" in df.columns else np.nan
        for _, s in seg_df.iterrows():
            for ax in [axs[4], axs[0]]:
                ax.axvspan(pd.to_datetime(s["t_start"]),
                           pd.to_datetime(s["t_end"]),
                           color=color, alpha=alpha)
            if np.isfinite(ymax):
                axs[4].text(pd.to_datetime(s["t_start"]), ymax * 0.995,
                            f"{int(s['duration_sec'])}s\n±{s['excursion_ft']:.0f} ft",
                            fontsize=7, color=color, rotation=90, va="top")

    # Date & time formatting
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(axs[-1].get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Activity (Ream+Backream, TFLO>10) — Blue strips on BPOS/TFLO",
                 y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_bpos_strip_blue(df: pd.DataFrame, seg_df: pd.DataFrame,
                         out_png: Path, color: str, alpha: float) -> Path:
    """Compact BPOS strip with blue shading and date+time axis."""
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    ax.plot(df["timestamp"], df["BPOS"], color="black", lw=1)
    ax.set_ylabel("BPOS (ft)")
    ax.set_title("BPOS with Activity Strips (Ream+Backream) — Blue")
    ax.grid(True, which="both", alpha=0.25)

    if not seg_df.empty:
        ymax = float(df["BPOS"].max(skipna=True))
        for _, s in seg_df.iterrows():
            ax.axvspan(pd.to_datetime(s["t_start"]), pd.to_datetime(s["t_end"]),
                       color=color, alpha=alpha)
            ax.text(pd.to_datetime(s["t_start"]), ymax * 0.995,
                    f"{int(s['duration_sec'])}s\n±{s['excursion_ft']:.0f} ft",
                    fontsize=7, color=color, rotation=90, va="top")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# ----------------------------
# Interactive HTML — Plotly via CDN (no python-plotly dependency)
# ----------------------------
def export_plotly_html(df: pd.DataFrame, seg_df: pd.DataFrame,
                       out_html: Path, color: str, alpha: float) -> Path:
    """
    Create an interactive HTML using Plotly via CDN (no local plotly package needed).
    Includes range slider & selector. Single BPOS trace + blue strips.
    """
    x = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    y = df["BPOS"].fillna(np.nan).tolist()

    shapes_js = []
    for _, s in seg_df.iterrows():
        shapes_js.append({
            "type": "rect",
            "xref": "x", "yref": "paper",
            "x0": pd.to_datetime(s["t_start"]).strftime("%Y-%m-%dT%H:%M:%S"),
            "x1": pd.to_datetime(s["t_end"]).strftime("%Y-%m-%dT%H:%M:%S"),
            "y0": 0, "y1": 1,
            "fillcolor": color,
            "opacity": alpha,
            "line": {"width": 0},
        })

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>BPOS Activity (Blue Strips)</title>
<script src.plot.ly/plotly-2.32.0.min.js</script>
<style>
  body {{ font-family: Arial, sans-serif; }}
  #chart {{ width: 100%; height: 80vh; }}
  .note {{ font-size: 12px; color: #555; margin: 8px 0 0 6px; }}
</style>
</head>
<body>
<div id='chart'></div>
<div class='note'>Zoom with mouse drag, pan with right-click/drag, use range slider & selector below. Double-click to reset.</div>
<script>
var x = {json.dumps(x)};
var y = {json.dumps(y)};
var data = [{{
  x: x,
  y: y,
  mode: 'lines',
  name: 'BPOS',
  line: {{color:'black', width:1}},
  hovertemplate: '%{{x}}<br>BPOS: %{{y:.2f}} ft'+'<extra></extra>'
}}];

var layout = {{
  title: 'BPOS with Activity Strips (Ream+Backream) — Blue',
  xaxis: {{
    title: 'Date & Time',
    type: 'date',
    rangeslider: {{visible: true}},
    rangeselector: {{
      buttons: [
        {{count: 1, label: '1h', step: 'hour', stepmode: 'backward'}},
        {{count: 6, label: '6h', step: 'hour', stepmode: 'backward'}},
        {{count: 1, label: '1d', step: 'day', stepmode: 'backward'}},
        {{step: 'all'}}
      ]
    }}
  }},
  yaxis: {{title: 'BPOS (ft)', fixedrange: false}},
  shapes: {json.dumps(shapes_js)},
  margin: {{l: 60, r: 20, t: 50, b: 60}},
}};

Plotly.newPlot('chart', data, layout, {{responsive: true}});
</script>
</body>
</html>"""

    out_html.write_text(html, encoding="utf-8")
    return out_html


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Combined Ream+Backream detection (blue strips) on BPOS."
    )
    p.add_argument("--excel", required=True, help="Path to the Excel historian (.xlsx)")
    p.add_argument("--outdir", default="", help="Output directory (default: alongside Excel)")
    p.add_argument("--tflo", type=float, default=DEFAULT_TFLO_ON_THRESHOLD, help="Pump ON threshold (gpm)")
    p.add_argument("--minsec", type=int, default=DEFAULT_MIN_ACTIVITY_SECONDS, help="Minimum segment duration (sec)")
    p.add_argument("--ream", type=float, default=DEFAULT_REAM_DELTA_FT, help="Ream excursion threshold (ft)")
    p.add_argument("--back", type=float, default=DEFAULT_BACK_DELTA_FT, help="Backream excursion threshold (ft)")
    p.add_argument("--color", default=STRIP_COLOR_DEFAULT, help="Strip color (hex)")
    p.add_argument("--alpha", type=float, default=STRIP_ALPHA_DEFAULT, help="Strip opacity (0–1)")
    return p.parse_args()


def main():
    args = parse_args()

    excel_path = Path(args.excel)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    # Output directory & base names
    outdir = Path(args.outdir) if args.outdir else excel_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    base = excel_path.stem

    # Load & process
    df = load_and_clean_excel(excel_path)
    seg_df = detect_activity_segments(
        df,
        tflo_thr=args.tflo,
        min_sec=args.minsec,
        ream_ft=args.ream,
        back_ft=args.back,
    )

    # Save segments
    seg_csv = outdir / f"{base}_segments_blue.csv"
    seg_df.to_csv(seg_csv, index=False)
    print(f"[CSV] Saved segments → {seg_csv} (n={len(seg_df)})")

    # Static charts
    overview_png = outdir / f"{base}_overview_blue.png"
    bpos_png     = outdir / f"{base}_bpos_strip_blue.png"
    plot_overview_with_blue_strips(df, seg_df, overview_png, args.color, args.alpha)
    plot_bpos_strip_blue(df, seg_df, bpos_png, args.color, args.alpha)
    print(f"[PNG] Saved overview  → {overview_png}")
    print(f"[PNG] Saved BPOS strip → {bpos_png}")

    # Interactive HTML
    html_path = outdir / f"{base}_interactive_blue.html"
    export_plotly_html(df, seg_df, html_path, args.color, args.alpha)
    print(f"[HTML] Saved interactive chart → {html_path}")
    print("Done.")


if __name__ == "__main__":
    main()