#!/usr/bin/env python3
"""Interactive Set A pool-size HTML: slider + play + ez_cutoff buttons.

Custom JS owns pool/cutoff/play state so the three controls stay in sync
(Plotly frames cannot drive a cutoff toggle during animation).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_POOL,
    EZ_CUTOFFS,
    FF_MIN,
    MODES,
    NORMAL_COLOR,
    PURITY_MIN,
    SPECIAL_COLOR,
    TRISOMY_COLOR,
    ez_ratio_col,
)

console = Console()

STATUS_ORDER = ("Normal", "trisomy", "special")


def _is_trisomy(label: str) -> bool:
    return bool(re.match(r"^T\d", str(label)))


def _prep(df: pd.DataFrame, high_ff: bool) -> pd.DataFrame:
    out = df.copy()
    out["orig_sample"] = out["orig_sample"].astype(str)
    out["ff_before_mq"] = pd.to_numeric(out["ff_before_mq"], errors="coerce")
    out["purity"] = pd.to_numeric(out["purity"], errors="coerce")
    out = out.loc[out["ff_before_mq"].notna()].copy()
    if high_ff:
        out = out.loc[out["ff_before_mq"] >= FF_MIN].copy()
    else:
        out = out.loc[out["ff_before_mq"] < FF_MIN].copy()
    out["is_trisomy"] = out["label"].map(_is_trisomy)
    out["is_special"] = pd.to_numeric(out["purity"], errors="coerce") < PURITY_MIN
    out["status"] = np.where(
        out["is_special"],
        "special",
        np.where(out["is_trisomy"], "trisomy", "Normal"),
    )
    return out


def _pack_status(sub: pd.DataFrame, y_col: str) -> dict:
    pur = pd.to_numeric(sub["purity"], errors="coerce")
    custom = [
        [
            str(s),
            str(lab),
            None if pd.isna(p) else round(float(p), 4),
        ]
        for s, lab, p in zip(sub["set"].astype(str), sub["label"].astype(str), pur)
    ]
    return {
        "x": [None if pd.isna(v) else float(v) for v in sub["ff_before_mq"]],
        "y": [None if pd.isna(v) else float(v) for v in sub[y_col]],
        "text": sub["orig_sample"].astype(str).tolist(),
        "custom": custom,
    }


def _block(df: pd.DataFrame, y_col: str) -> dict:
    n_t = int(df["is_trisomy"].sum())
    n_n = int((~df["is_trisomy"]).sum())
    packed = {st: _pack_status(df.loc[df["status"] == st], y_col) for st in STATUS_ORDER}
    packed["n_normal"] = n_n
    packed["n_trisomy"] = n_t
    packed["n_special"] = int(df["is_special"].sum())
    return packed


def _load_pools(sweep_dir: Path) -> tuple[list[int], dict[int, pd.DataFrame]]:
    by_pool: dict[int, pd.DataFrame] = {}
    for pdir in sorted(sweep_dir.glob("pool_*")):
        tsv = pdir / "abnormality_signal_ratio.tsv"
        if not tsv.is_file():
            continue
        try:
            pool = int(pdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        by_pool[pool] = pd.read_csv(tsv, sep="\t")
    pools = sorted(by_pool)
    if not pools:
        raise click.ClickException(f"No pool_*/abnormality_signal_ratio.tsv under {sweep_dir}")
    return pools, by_pool


def build_payload(
    by_pool: dict[int, pd.DataFrame],
    pools: list[int],
    high_ff: bool,
) -> dict:
    out: dict = {}
    for pool in pools:
        raw = by_pool[pool]
        df = _prep(raw, high_ff=high_ff)
        out[str(pool)] = {}
        for cut in EZ_CUTOFFS:
            col = ez_ratio_col(cut)
            if col not in df.columns:
                raise click.ClickException(f"missing {col} in pool_{pool}")
            out[str(pool)][f"{cut:g}"] = _block(df, col)
    return out


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>__TITLE_TEXT__</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 16px 24px 32px; color: #222; }
    #plot { width: 1050px; max-width: 100%; height: 600px; }
    .controls { width: 1050px; max-width: 100%; margin-top: 8px; }
    .row { display: flex; align-items: center; gap: 12px; margin: 10px 0; flex-wrap: wrap; }
    .label { font-size: 14px; color: #333; min-width: 90px; }
    #pool-slider { flex: 1; min-width: 240px; }
    #pool-value { font-variant-numeric: tabular-nums; min-width: 3ch; }
    button {
      font-size: 13px; padding: 6px 12px; border: 1px solid #bbb; background: #fff;
      border-radius: 4px; cursor: pointer;
    }
    button:hover { background: #f4f4f4; }
    button.active {
      background: #1f77b4; color: #fff; border-color: #1f77b4;
    }
    button.play { min-width: 92px; }
  </style>
</head>
<body>
  <div id="plot"></div>
  <div class="controls">
    <div class="row">
      <button id="btn-play" class="play" type="button">▶ Play</button>
      <span class="label">pool size: <strong id="pool-value"></strong></span>
      <input id="pool-slider" type="range" min="0" step="1"/>
    </div>
    <div class="row">
      <span class="label">ez_cutoff</span>
      <button id="btn-cut-3" type="button" data-cut="3">cutoff = 3</button>
      <button id="btn-cut-45" type="button" data-cut="4.5">cutoff = 4.5</button>
    </div>
  </div>
  <script type="application/json" id="plot-data">__PAYLOAD__</script>
  <script>
  (function () {
    const payload = JSON.parse(document.getElementById("plot-data").textContent);
    const pools = payload.pools;
    const mode = payload.mode;
    const ffTag = payload.ff_tag;
    const defaultPool = payload.default_pool;
    const gd = document.getElementById("plot");
    const slider = document.getElementById("pool-slider");
    const poolValue = document.getElementById("pool-value");
    const btnPlay = document.getElementById("btn-play");
    const btn3 = document.getElementById("btn-cut-3");
    const btn45 = document.getElementById("btn-cut-45");

    const state = {
      poolIdx: Math.max(0, pools.indexOf(defaultPool)),
      cutoff: "3",
      timer: null,
    };
    slider.max = String(pools.length - 1);
    slider.value = String(state.poolIdx);

    const marker = {
      Normal:  { color: "__NORMAL__", size: 8, opacity: 0.55, symbol: "circle" },
      trisomy: { color: "__TRISOMY__", size: 9, opacity: 0.92, symbol: "circle" },
      special: { color: "__SPECIAL__", size: 10, opacity: 0.95, symbol: "diamond" },
    };
    const names = {
      Normal: "Normal",
      trisomy: "trisomy",
      special: "purity<0.8",
    };

    function block() {
      const pool = String(pools[state.poolIdx]);
      return payload.by_pool[pool][state.cutoff];
    }

    function titleText() {
      const pool = pools[state.poolIdx];
      const half = pool / 2;
      const b = block();
      const cut = state.cutoff === "4.5" ? "4.5" : "3";
      return mode + " SetA " + ffTag + " · ref " + half + "+" + half
        + " · ez_cutoff=" + cut + " · N=" + b.n_normal + ", T=" + b.n_trisomy;
    }

    function traces() {
      const b = block();
      return ["Normal", "trisomy", "special"].map(function (st) {
        const d = b[st];
        return {
          type: "scatter",
          mode: "markers",
          name: names[st],
          legendgroup: st,
          x: d.x,
          y: d.y,
          text: d.text,
          customdata: d.custom,
          marker: marker[st],
          hovertemplate:
            "%{text}<br>set=%{customdata[0]} label=%{customdata[1]}"
            + "<br>ff=%{x:.4f}<br>purity=%{customdata[2]}"
            + "<br>ratio=%{y:.4f}<extra>" + names[st] + "</extra>",
        };
      });
    }

    const layout = {
      title: { text: titleText() },
      xaxis: {
        title: "ff_before_mq",
        tickformat: ".0%",
        showgrid: true,
        gridcolor: "rgba(0,0,0,0.06)",
      },
      yaxis: {
        title: "ezscore signal ratio",
        range: [-0.02, 1.05],
        showgrid: true,
        gridcolor: "rgba(0,0,0,0.06)",
      },
      template: "plotly_white",
      height: 600,
      width: 1050,
      showlegend: true,
      legend: {
        orientation: "v",
        yanchor: "top",
        y: 1.0,
        xanchor: "left",
        x: 1.02,
        bgcolor: "rgba(255,255,255,0.9)",
        bordercolor: "rgba(0,0,0,0.12)",
        borderwidth: 1,
      },
      margin: { t: 80, b: 60, r: 180, l: 60 },
      plot_bgcolor: "rgba(248,249,250,1)",
    };

    function syncButtons() {
      poolValue.textContent = String(pools[state.poolIdx]);
      slider.value = String(state.poolIdx);
      btn3.classList.toggle("active", state.cutoff === "3");
      btn45.classList.toggle("active", state.cutoff === "4.5");
      btnPlay.textContent = state.timer ? "❚❚ Pause" : "▶ Play";
    }

    function render() {
      layout.title = { text: titleText() };
      Plotly.react(gd, traces(), layout, { responsive: true, displaylogo: false });
      syncButtons();
    }

    function stopPlay() {
      if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
      }
      syncButtons();
    }

    function startPlay() {
      if (state.timer) return;
      state.timer = setInterval(function () {
        state.poolIdx = (state.poolIdx + 1) % pools.length;
        render();
      }, 280);
      syncButtons();
    }

    btnPlay.addEventListener("click", function () {
      if (state.timer) stopPlay();
      else startPlay();
    });
    slider.addEventListener("input", function () {
      state.poolIdx = parseInt(slider.value, 10);
      render();
    });
    btn3.addEventListener("click", function () {
      state.cutoff = "3";
      render();
    });
    btn45.addEventListener("click", function () {
      state.cutoff = "4.5";
      render();
    });

    render();
  })();
  </script>
</body>
</html>
"""


def write_html(
    payload: dict,
    out_html: Path,
    mode: str,
    ff_tag: str,
    title_text: str,
) -> None:
    blob = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    html = (
        HTML_TEMPLATE.replace("__PAYLOAD__", blob)
        .replace("__TITLE_TEXT__", title_text)
        .replace("__NORMAL__", NORMAL_COLOR)
        .replace("__TRISOMY__", TRISOMY_COLOR)
        .replace("__SPECIAL__", SPECIAL_COLOR)
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)
    console.print(f"[green]wrote[/green] {out_html} ({out_html.stat().st_size} bytes)")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--result-root",
    default=str(DEFAULT_OUT),
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--mode", default="all", show_default=True, type=click.Choice(["all", "modeA", "modeB"]))
def main(result_root: Path, mode: str) -> None:
    modes = list(MODES) if mode == "all" else [mode]
    plot_dir = result_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for m in modes:
        sweep_dir = result_root / m
        pools, by_pool = _load_pools(sweep_dir)
        default_pool = DEFAULT_POOL if DEFAULT_POOL in pools else pools[len(pools) // 2]
        for high, tag, ff_label in (
            (True, "ff_ge_1pct", "ff≥1%"),
            (False, "ff_lt_1pct", "ff<1%"),
        ):
            probe = _prep(by_pool[default_pool], high_ff=high)
            if probe.empty:
                stale = plot_dir / f"{m}_SetA_{tag}_pool_size.html"
                if stale.is_file():
                    stale.unlink()
                    console.print(f"[yellow]removed empty[/yellow] {stale}")
                continue
            by = build_payload(by_pool, pools, high_ff=high)
            payload = {
                "mode": m,
                "ff_tag": ff_label,
                "pools": pools,
                "default_pool": default_pool,
                "by_pool": by,
            }
            out = plot_dir / f"{m}_SetA_{tag}_pool_size.html"
            write_html(payload, out, m, ff_label, f"{m} SetA {ff_label} pool size")


if __name__ == "__main__":
    main()
