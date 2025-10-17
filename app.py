# -*- coding: utf-8 -*-
"""
Combined Activity Detection (Reaming + Backreaming, single blue color)
Generates OFFLINE interactive HTML (Plotly JS inlined), PNGs, and CSV.

Rules:
  Pump ON            : TFLO > TFLO_ON_THRESHOLD (default 10 gpm)
  Excursion (combined, OR):
      (DBTM - CDEPTH) >= +REAM_DELTA_FT          OR
      (CDEPTH - DBTM) >= BACK_DELTA_FT
  Minimum Duration   : MIN_ACTIVITY_SECONDS (default 10 s)

Outputs (written to --outdir, default ./site):
  - <base>_segments_blue.csv
  - <base>_overview_blue.png
  - <base>_bpos_strip_blue.png
  - <base>_interactive_blue_offline.html  (self-contained)
  - index.html  (copy of offline HTML for convenience)

Usage:
  python app.py --excel "time_12.25 inch HS.xlsx" --outdir "site"

Optional flags:
  --tflo 10 --minsec 10 --ream 3 --back 20 --color "#1f77b4" --alpha 0.22
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go

# ----------------------------
# Defaults (override via CLI)
# ----------------------------
DEFAULT_TFLO_ON_THRESHOLD     = 10.0   # gpm
DEFAULT_MIN_ACTIVITY_SECONDS  = 10     # s
DEFAULT_REAM_DELTA_FT         = 3.0    # ft
DEFAULT_BACK_DELTA_FT         = 20.0   # ft
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

    # Timestamp from Date or Time or first column
    if "Date" in df.columns:
        ts = pd.to_datetime(df["Date"], errors="coerce")
    elif "Time" in df.columns:
        ts = pd.to_datetime(df["Time"], errors="coerce")
    else:
        ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df["timestamp"] = ts

    # Replace sentinel -> NaN and coerce numerics
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
    Returns columns: t_start, t_end, duration_sec, excursion_ft.
    """
    # Required channels
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

        # Edge handling
        if mask.iloc[0]:
            starts = [mask.index[0]] + starts
        if mask.iloc[-1]:
            ends = ends + [mask.index[-1]]

        for s, e in zip(starts, ends):
            t0 = df.loc[s, "timestamp"]
            t1 = df.loc[e, "timestamp"]
            dur_sec = (t1 - t0).total_seconds()

            if dur_sec >= min_sec:
                dsub = (df["DBTM"] - df["CDEPTH"]).iloc[s:e+1].values
                excursion_ft = float(np.nanmax(np.maximum(dsub, -dsub)))  # max |delta|
                segments.append({
                    "t_start": t0,
                    "t_end"  : t1,
                    "duration_sec": dur_sec,
                    "excursion_ft": excursion_ft,
                })

    return pd.DataFrame(segments).sort_values("t_start").reset_index(drop=True)


# ----------------------------
# Matplotlib plots
# ----------------------------
def plot_overview_with_blue_strips(df: pd.DataFrame, seg_df: pd.DataFrame,
                                   out_png: Path, color: str, alpha: float) -> Path:
    """5‑panel overview (TFLO, SPPA, ECD, ROP5, BPOS) with blue strips on BPOS & TFLO."""
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

    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(axs[-1].get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Activity (Ream+Backream, TFLO>10) — Blue strips on BPOS/TFLO", y=0.995, fontsize=13)
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
# Plotly OFFLINE HTML (self-contained)
# ----------------------------
def export_plotly_html_offline(df: pd.DataFrame, seg_df: pd.DataFrame,
                               out_html: Path, color: str, alpha: float) -> Path:
    """Create a fully self-contained HTML (Plotly JS inlined) for offline use."""
    fig = go.Figure()

    # BPOS trace
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["BPOS"],
        mode="lines", name="BPOS",
        line=dict(color="black", width=1),
        hovertemplate="%{x}<br>BPOS: %{y:.2f} ft<extra></extra>"
    ))

    # Blue strips as shapes
    for _, s in seg_df.iterrows():
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=pd.to_datetime(s["t_start"]), x1=pd.to_datetime(s["t_end"]),
            y0=0, y1=1, fillcolor=color, opacity=alpha, line=dict(width=0)
        )

    fig.update_layout(
        title="BPOS with Activity Strips (Ream+Backream) — Blue (Offline)",
        xaxis=dict(
            title="Date & Time",
            type="date",
            rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all"),
            ]),
        ),
        yaxis=dict(title="BPOS (ft)", fixedrange=False),
        margin=dict(l=60, r=20, t=50, b=60),
    )

    # Inline Plotly JS
    fig.write_html(str(out_html), include_plotlyjs="inline", full_html=True)
    return out_html


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Combined Ream+Backream detection (blue strips) on BPOS, OFFLINE HTML."
    )
    p.add_argument("--excel", required=True, help="Path to the Excel historian (.xlsx)")
    p.add_argument("--outdir", default="site", help="Output directory (default: ./site)")
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

    # Output dir
    outdir = Path(args.outdir)
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

    # OFFLINE interactive HTML
    html_offline = outdir / f"{base}_interactive_blue_offline.html"
    export_plotly_html_offline(df, seg_df, html_offline, args.color, args.alpha)
    print(f"[HTML] Saved OFFLINE interactive chart → {html_offline}")

    # Make index.html (landing file)
    index_html = outdir / "index.html"
    index_html.write_text(html_offline.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[HTML] Copied offline chart → {index_html}")

    print("Done.")


if __name__ == "__main__":
    main()
