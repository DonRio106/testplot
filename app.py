# -*- coding: utf-8 -*-
"""
Combined Activity Detection (Reaming + Backreaming, single blue color)
Generates OFFLINE interactive HTML (Plotly JS inlined), PNGs, and CSV.

This is a revised version with:
 - Fixes to segment edge handling (start/end indices and duration calculation).
 - Robust handling of CSV or Excel input.
 - Better handling when required columns are missing.
 - Logging instead of print for easier debugging.
 - Minor defensive checks in plotting.
 - Type hints for key functions.

Usage:
  python app.py --excel "time_12.25 inch HS.xlsx" --outdir "site"
"""
from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("activity_detection")

# ----------------------------
# Data loading & cleaning
# ----------------------------
def load_and_clean(path: Path) -> pd.DataFrame:
    """Load historian Excel (.xlsx) or CSV, build timestamp, replace sentinels, coerce numerics."""
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls", ".xlsm"]:
        df = pd.read_excel(path, engine="openpyxl")
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported input format. Provide .xlsx/.xls/.csv")

    df.columns = [str(c).strip() for c in df.columns]

    # Timestamp heuristics: prefer 'Date' or 'Time' or first column
    if "Date" in df.columns:
        ts = pd.to_datetime(df["Date"], errors="coerce")
    elif "Time" in df.columns:
        ts = pd.to_datetime(df["Time"], errors="coerce")
    else:
        ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Replace sentinel -> NaN and coerce numerics for expected columns
    df = df.replace(SENTINEL, np.nan)
    for c in ["TFLO", "SPPA", "ECD", "ROP5", "DBTM", "BPOS", "TVDE", "CDEPTH", "RPM", "TQA"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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

    Corrected segmentation:
      - Find indices where mask is True.
      - Group contiguous indices into segments.
      - Use the first and last True timestamps for t_start and t_end.
      - Compute duration as t_end - t_start (seconds).
      - Compute excursion as max(|DBTM - CDEPTH|) within the True range.

    Returns DataFrame with columns: t_start, t_end, duration_sec, excursion_ft.
    """
    # Required channels
    for col in ["TFLO", "DBTM", "CDEPTH"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    pump_on = (df["TFLO"] > tflo_thr).fillna(False).to_numpy()
    delta   = (df["DBTM"] - df["CDEPTH"]).to_numpy()

    # condition: ream OR backream
    cond_any = (delta >= ream_ft) | ((-delta) >= back_ft)

    # mask of valid activity (pump on, excursion, and not-NaN)
    valid_db = ~np.isnan(df["DBTM"].to_numpy())
    valid_cd = ~np.isnan(df["CDEPTH"].to_numpy())
    mask = pump_on & cond_any & valid_db & valid_cd

    segments = []
    true_idxs = np.nonzero(mask)[0]

    if true_idxs.size == 0:
        logger.info("No activity segments detected (mask empty).")
        return pd.DataFrame(segments, columns=["t_start", "t_end", "duration_sec", "excursion_ft"])

    # find contiguous groups in true_idxs
    breaks = np.where(np.diff(true_idxs) > 1)[0]
    group_starts = np.concatenate(([0], breaks + 1))
    group_ends = np.concatenate((breaks, [true_idxs.size - 1]))

    for gs, ge in zip(group_starts, group_ends):
        s_idx = int(true_idxs[gs])
        e_idx = int(true_idxs[ge])  # last index where mask is True in this group

        t0 = pd.to_datetime(df.loc[s_idx, "timestamp"])
        t1 = pd.to_datetime(df.loc[e_idx, "timestamp"])
        dur_sec = (t1 - t0).total_seconds()
        if dur_sec < 0:
            # defensive: skip if timestamps out-of-order for some reason
            logger.warning("Negative duration encountered for indices %s-%s; skipping.", s_idx, e_idx)
            continue

        if dur_sec >= min_sec:
            dsub = delta[s_idx:e_idx + 1]
            # excursion is max absolute delta
            excursion_ft = float(np.nanmax(np.abs(dsub)))
            segments.append({
                "t_start": t0,
                "t_end"  : t1,
                "duration_sec": dur_sec,
                "excursion_ft": excursion_ft,
            })
        else:
            logger.debug("Segment %s-%s rejected (duration %ds < min %ds)", s_idx, e_idx, dur_sec, min_sec)

    seg_df = pd.DataFrame(segments).sort_values("t_start").reset_index(drop=True)
    logger.info("Detected %d segments (after duration filter).", len(seg_df))
    return seg_df

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
        else:
            ax.text(0.5, 0.5, f"No {label} data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            ax.set_ylabel(label)

    panel(axs[0], "TFLO", "teal",   "TFLO (gpm)")
    panel(axs[1], "SPPA", "crimson","SPPA (psi)")
    panel(axs[2], "ECD",  "purple", "ECD (ppg)")
    panel(axs[3], "ROP5", "navy",   "ROP5 (ft/h)")
    panel(axs[4], "BPOS", "black",  "BPOS (ft)")
    axs[-1].set_xlabel("Date & Time")

    if not seg_df.empty:
        # determine a y-position for annotations if BPOS exists; fallback to 0.99 in paper coords
        if "BPOS" in df.columns and df["BPOS"].notna().any():
            ymax = float(df["BPOS"].max(skipna=True))
            text_coord_y = ymax * 0.995
            use_data_coords = True
        else:
            text_coord_y = 0.99
            use_data_coords = False

        for _, s in seg_df.iterrows():
            for ax in [axs[4], axs[0]]:
                ax.axvspan(pd.to_datetime(s["t_start"]),
                           pd.to_datetime(s["t_end"]),
                           color=color, alpha=alpha)
            # annotation
            if use_data_coords:
                axs[4].text(pd.to_datetime(s["t_start"]), text_coord_y,
                            f"{int(s['duration_sec'])}s\n±{s['excursion_ft']:.0f} ft",
                            fontsize=7, color=color, rotation=90, va="top")
            else:
                # place in axes fraction coordinates
                axs[4].annotate(f"{int(s['duration_sec'])}s\n±{s['excursion_ft']:.0f} ft",
                                xy=(pd.to_datetime(s["t_start"]), 0.99),
                                xycoords=("data", "axes fraction"),
                                fontsize=7, color=color, rotation=90, va="top")

    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(axs[-1].get_xticklabels(), rotation=30, ha="right")

    fig.suptitle(f"Activity (Ream+Backream, TFLO>{DEFAULT_TFLO_ON_THRESHOLD}) — Blue strips on BPOS/TFLO", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("Saved overview PNG → %s", out_png)
    return out_png


def plot_bpos_strip_blue(df: pd.DataFrame, seg_df: pd.DataFrame,
                         out_png: Path, color: str, alpha: float) -> Path:
    """Compact BPOS strip with blue shading and date+time axis."""
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    if "BPOS" in df.columns and df["BPOS"].notna().any():
        ax.plot(df["timestamp"], df["BPOS"], color="black", lw=1)
    else:
        ax.text(0.5, 0.5, "No BPOS data available", transform=ax.transAxes,
                ha="center", va="center", color="gray")

    ax.set_ylabel("BPOS (ft)")
    ax.set_title("BPOS with Activity Strips (Ream+Backream) — Blue")
    ax.grid(True, which="both", alpha=0.25)

    if not seg_df.empty:
        if "BPOS" in df.columns and df["BPOS"].notna().any():
            ymax = float(df["BPOS"].max(skipna=True))
            text_y = ymax * 0.995
            use_data_coords = True
        else:
            text_y = 0.99
            use_data_coords = False

        for _, s in seg_df.iterrows():
            ax.axvspan(pd.to_datetime(s["t_start"]), pd.to_datetime(s["t_end"]),
                       color=color, alpha=alpha)
            if use_data_coords:
                ax.text(pd.to_datetime(s["t_start"]), text_y,
                        f"{int(s['duration_sec'])}s\n±{s['excursion_ft']:.0f} ft",
                        fontsize=7, color=color, rotation=90, va="top")
            else:
                ax.annotate(f"{int(s['duration_sec'])}s\n±{s['excursion_ft']:.0f} ft",
                            xy=(pd.to_datetime(s["t_start"]), 0.99),
                            xycoords=("data", "axes fraction"),
                            fontsize=7, color=color, rotation=90, va="top")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("Saved BPOS strip PNG → %s", out_png)
    return out_png

# ----------------------------
# Plotly OFFLINE HTML (self-contained)
# ----------------------------
def export_plotly_html_offline(df: pd.DataFrame, seg_df: pd.DataFrame,
                               out_html: Path, color: str, alpha: float) -> Path:
    """
    Create a fully self-contained HTML (Plotly JS inlined) for offline use.

    Layout:
      Row 1: TFLO (primary Y) and SPPA (secondary Y)
      Row 2: BPOS (single axis)
      Row 3: RPM (primary Y) and TQA (secondary Y)

    Missing columns are handled gracefully (an empty trace is added so layout/legend remains consistent).
    Blue activity strips are drawn across the full figure (yref='paper').
    """
    # Prepare subplot layout with secondary y for rows 1 and 3
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": True}]
        ],
        row_heights=[0.3, 0.4, 0.3],
    )

    # Helper to safe-get series or empty
    def series_or_empty(name):
        if name in df.columns and df[name].notna().any():
            return df["timestamp"], df[name]
        else:
            return [], []

    # Row 1: TFLO (primary) and SPPA (secondary)
    x_tflo, y_tflo = series_or_empty("TFLO")
    x_sppa, y_sppa = series_or_empty("SPPA")

    fig.add_trace(go.Scatter(
        x=x_tflo, y=y_tflo,
        mode="lines", name="TFLO (gpm)",
        line=dict(color="teal", width=1),
        hovertemplate="%{x}<br>TFLO: %{y:.2f} gpm<extra></extra>"
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x_sppa, y=y_sppa,
        mode="lines", name="SPPA (psi)",
        line=dict(color="crimson", width=1),
        hovertemplate="%{x}<br>SPPA: %{y:.2f} psi<extra></extra>"
    ), row=1, col=1, secondary_y=True)

    # Row 2: BPOS
    x_bpos, y_bpos = series_or_empty("BPOS")
    fig.add_trace(go.Scatter(
        x=x_bpos, y=y_bpos,
        mode="lines", name="BPOS (ft)",
        line=dict(color="black", width=1),
        hovertemplate="%{x}<br>BPOS: %{y:.2f} ft<extra></extra>"
    ), row=2, col=1)

    # Row 3: RPM (primary) and TQA (secondary)
    x_rpm, y_rpm = series_or_empty("RPM")
    x_tqa, y_tqa = series_or_empty("TQA")

    fig.add_trace(go.Scatter(
        x=x_rpm, y=y_rpm,
        mode="lines", name="RPM",
        line=dict(color="navy", width=1),
        hovertemplate="%{x}<br>RPM: %{y:.2f}<extra></extra>"
    ), row=3, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x_tqa, y=y_tqa,
        mode="lines", name="TQA",
        line=dict(color="purple", width=1, dash="dot"),
        hovertemplate="%{x}<br>TQA: %{y:.2f}<extra></extra>"
    ), row=3, col=1, secondary_y=True)

    # Blue strips as shapes across the full figure
    for _, s in seg_df.iterrows():
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=pd.to_datetime(s["t_start"]), x1=pd.to_datetime(s["t_end"]),
            y0=0, y1=1, fillcolor=color, opacity=alpha, line=dict(width=0)
        )

    # Axis titles
    fig.update_yaxes(title_text="TFLO (gpm)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="SPPA (psi)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="BPOS (ft)", row=2, col=1)
    fig.update_yaxes(title_text="RPM", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="TQA", row=3, col=1, secondary_y=True)

    fig.update_layout(
        title="Activity (Ream+Backream) — TFLO/SPPA | BPOS | RPM/TQA (Offline)",
        xaxis=dict(title="Date & Time", type="date"),
        xaxis2=dict(title="Date & Time", type="date"),
        xaxis3=dict(title="Date & Time", type="date"),
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )

    # Range selector / slider applied to xaxis (the bottom-most xaxis is xaxis3 by default; use shared_xaxes)
    fig.update_layout(
        xaxis3=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all"),
            ])
        )
    )

    # Inline Plotly JS
    fig.write_html(str(out_html), include_plotlyjs="inline", full_html=True)
    logger.info("Saved offline interactive HTML → %s", out_html)
    return out_html

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Combined Ream+Backream detection (blue strips) on BPOS, OFFLINE HTML."
    )
    p.add_argument("--excel", required=True, help="Path to the Excel/CSV historian (.xlsx/.csv)")
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
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = excel_path.stem

    df = load_and_clean(excel_path)

    try:
        seg_df = detect_activity_segments(
            df,
            tflo_thr=args.tflo,
            min_sec=args.minsec,
            ream_ft=args.ream,
            back_ft=args.back,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return

    # Save segments (even if empty)
    seg_csv = outdir / f"{base}_segments_blue.csv"
    seg_df.to_csv(seg_csv, index=False)
    logger.info("Saved segments CSV → %s (n=%d)", seg_csv, len(seg_df))

    # Static charts
    overview_png = outdir / f"{base}_overview_blue.png"
    bpos_png     = outdir / f"{base}_bpos_strip_blue.png"
    plot_overview_with_blue_strips(df, seg_df, overview_png, args.color, args.alpha)
    plot_bpos_strip_blue(df, seg_df, bpos_png, args.color, args.alpha)

    # OFFLINE interactive HTML
    html_offline = outdir / f"{base}_interactive_blue_offline.html"
    export_plotly_html_offline(df, seg_df, html_offline, args.color, args.alpha)

    # Make index.html (landing file)
    index_html = outdir / "index.html"
    index_html.write_text(html_offline.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("Copied offline chart to → %s", index_html)

    logger.info("Done.")

if __name__ == "__main__":
    main()
