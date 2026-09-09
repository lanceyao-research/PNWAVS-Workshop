#!/usr/bin/env python3
"""Reaction-screening experiment simulator for SciLink's BO loop.

A richer cousin of MRSTutorial_MT01/BOAgent: instead of a single peak, the
"reaction" produces TWO competing species — a desired PRODUCT and an unwanted
BYPRODUCT — each giving a resolvable UV-Vis peak. The optimization target is the
**selectivity** = product_peak_area / byproduct_peak_area.

Ground truth (two control knobs, temperature_C and pH):
  * the product forms along a CURVED ridge in (T, pH) — pH-optimum rises with T
    (a banana-shaped valley), with a single best temperature along the ridge;
  * the byproduct grows at high pH and high T (hydrolysis / decomposition),
  * so the selectivity optimum is interior and OFF-CENTER, and `init` deliberately
    seeds the campaign in a far low-T / low-pH corner so you can watch BO travel.

Two emit modes, matching the two SciLink routing paths:
  * --mode spectra : writes wavelength_nm,absorbance CSVs (a measurement *curve*)
                     -> meta routes to ANALYSIS (multi-peak curve fit) -> feature
                     table -> PLANNING (scalarizer -> BO).
  * --mode tidy    : appends rows to results.csv (temperature_C, pH,
                     product_area, byproduct_area) -> meta routes straight to
                     PLANNING (scalarizer computes the ratio -> BO).

Conditions / metadata are written to a human-readable experiment.txt (NOT sidecar
JSON): global context + parameter bounds + objective, and — in spectra mode — a
per-file conditions manifest (filename, temperature_C, pH).

CLI
  python simulate.py init --mode spectra --out ./spectra_demo [--n 8]
  python simulate.py init --mode tidy    --out ./tidy_demo    [--n 8]
  python simulate.py run  --mode spectra --out ./spectra_demo \
      --params '{"temperature_C": 65, "pH": 9}'
  python simulate.py run  --mode tidy    --out ./tidy_demo \
      --params '[{"temperature_C": 65, "pH": 9}, {"temperature_C": 70, "pH": 8.5}]'
  python simulate.py truth                      # report the ground-truth optimum
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- configuration
BOUNDS = {"temperature_C": (5.0, 100.0), "pH": (1.0, 14.0)}
WAVELENGTHS = np.linspace(380, 700, 321)          # visible UV-Vis window (nm)

PRODUCT_LAMBDA = 540.0      # product absorbs green-yellow
BYPRODUCT_LAMBDA = 430.0    # byproduct absorbs blue
PEAK_WIDTH = 34.0           # nm (sigma)
# Scales the dimensionless yield functions into integrated peak areas large
# enough that spectrum peak HEIGHTS (= area / (width*sqrt(2pi))) sit well above
# baseline (~0.02) and noise (~0.006). A common factor on both species leaves
# the selectivity ratio — and therefore the optimum — unchanged.
SIGNAL_SCALE = 120.0

# init batch lives here — a far corner, away from the optimum
INIT_T_RANGE = (15.0, 38.0)
INIT_PH_RANGE = (2.5, 5.5)


# ---------------------------------------------------------------- ground truth
def _ridge_center_pH(temp):
    """Curved ridge: the pH that maximizes product formation rises with T."""
    return 4.3 + 0.0016 * (temp - 15.0) ** 2


def product_area(temp, pH):
    """Desired-product peak area: a banana ridge in (T, pH) with a single best T."""
    ridge = np.exp(-0.5 * ((pH - _ridge_center_pH(temp)) / 1.3) ** 2)
    along = np.exp(-0.5 * ((temp - 68.0) / 22.0) ** 2)
    return SIGNAL_SCALE * (1.0 * ridge * along + 0.04)


def byproduct_area(temp, pH):
    """Unwanted byproduct: grows at high pH (hydrolysis) and high T (decomp)."""
    hydrolysis = 0.55 / (1.0 + np.exp(-(pH - 9.5) / 1.2))
    decomp = 0.30 / (1.0 + np.exp(-(temp - 80.0) / 8.0))
    return SIGNAL_SCALE * (0.18 + hydrolysis + decomp)


def selectivity(temp, pH):
    """The optimization target."""
    return product_area(temp, pH) / byproduct_area(temp, pH)


def ground_truth_optimum(n=400):
    ts = np.linspace(*BOUNDS["temperature_C"], n)
    ps = np.linspace(*BOUNDS["pH"], n)
    TT, PP = np.meshgrid(ts, ps)
    S = selectivity(TT, PP)
    i = np.unravel_index(np.argmax(S), S.shape)
    return float(TT[i]), float(PP[i]), float(S[i])


# ---------------------------------------------------------------- spectrum model
def generate_spectrum(temp, pH, rng):
    """Two-peak UV-Vis spectrum whose peak AREAS encode product/byproduct amounts.

    Heights are set so a curve fit recovers the ground-truth areas; small pH
    solvatochromic shifts, a sloping baseline, run-to-run height variability and
    detector noise keep it realistic (and give the GP meaningful residuals)."""
    a_p = product_area(temp, pH) * (1.0 + rng.normal(0, 0.03))
    a_b = byproduct_area(temp, pH) * (1.0 + rng.normal(0, 0.03))
    # area = height * sigma * sqrt(2 pi)  ->  height = area / (sigma sqrt(2pi))
    norm = PEAK_WIDTH * np.sqrt(2 * np.pi)
    lam_p = PRODUCT_LAMBDA + 0.8 * (pH - 7.0)        # mild solvatochromic shift
    lam_b = BYPRODUCT_LAMBDA - 0.5 * (pH - 7.0)
    prod = (a_p / norm) * np.exp(-0.5 * ((WAVELENGTHS - lam_p) / PEAK_WIDTH) ** 2)
    byp = (a_b / norm) * np.exp(-0.5 * ((WAVELENGTHS - lam_b) / (PEAK_WIDTH * 0.9)) ** 2)
    baseline = 0.02 + 0.00004 * (WAVELENGTHS - 380)
    noise = rng.normal(0, 0.006, WAVELENGTHS.size)
    return np.clip(prod + byp + baseline + noise, 0, None)


# ---------------------------------------------------------------- io helpers
def _clip(p):
    return (float(np.clip(p["temperature_C"], *BOUNDS["temperature_C"])),
            float(np.clip(p["pH"], *BOUNDS["pH"])))


def _label(temp, pH):
    return f"spectrum_T{temp:.0f}_pH{pH:.1f}".replace(".", "p")


def _init_conditions(n, seed):
    rng = np.random.default_rng(seed)
    # a small even-ish grid inside the far corner, plus jitter
    k = max(2, int(round(np.sqrt(n))))
    ts = np.linspace(*INIT_T_RANGE, k)
    ps = np.linspace(*INIT_PH_RANGE, max(2, int(np.ceil(n / k))))
    pts = [(float(t), float(p)) for t in ts for p in ps][:n]
    return [{"temperature_C": round(t + rng.normal(0, 0.6), 2),
             "pH": round(p + rng.normal(0, 0.12), 2)} for t, p in pts]


def _write_experiment_txt(out: Path, mode: str, manifest_rows):
    lines = [
        "EXPERIMENT METADATA",
        "===================",
        "Technique: UV-Vis absorption spectroscopy of a reaction mixture"
        if mode == "spectra" else
        "Technique: reaction screening (extracted peak areas per run)",
        "System: a desired PRODUCT and an unwanted BYPRODUCT form competitively;",
        "        each gives one UV-Vis peak (product ~540 nm, byproduct ~430 nm).",
        "",
        "OBJECTIVE: maximize selectivity = product_peak_area / byproduct_peak_area",
        "",
        "CONTROLLABLE PARAMETERS (and allowed instrument ranges):",
        f"  - temperature_C: {BOUNDS['temperature_C'][0]:g} to {BOUNDS['temperature_C'][1]:g}",
        f"  - pH:            {BOUNDS['pH'][0]:g} to {BOUNDS['pH'][1]:g}",
        "",
        "GOAL: recommend the next (temperature_C, pH) to run that maximizes selectivity.",
    ]
    if mode == "spectra" and manifest_rows:
        lines += [
            "",
            "PER-FILE CONDITIONS (filename, temperature_C, pH):",
            *[f"  {fn}, {t:g}, {p:g}" for fn, t, p in manifest_rows],
        ]
    (out / "experiment.txt").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- spectra mode
def _save_spectrum(out: Path, temp, pH, rng):
    label = _label(temp, pH)
    absb = generate_spectrum(temp, pH, rng)
    import csv
    with open(out / f"{label}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wavelength_nm", "absorbance"])
        for wl, a in zip(WAVELENGTHS, absb):
            w.writerow([f"{wl:.2f}", f"{a:.5f}"])
    return f"{label}.csv"


def _rebuild_manifest(out: Path):
    """Read conditions.json (the running record) -> manifest rows for experiment.txt."""
    rec = json.loads((out / "conditions.json").read_text()) if (out / "conditions.json").is_file() else {}
    rows = [(fn, c["temperature_C"], c["pH"]) for fn, c in rec.items()]
    return sorted(rows)


def cmd_spectra_init(out: Path, n, seed):
    rng = np.random.default_rng(seed)
    rec = {}
    for c in _init_conditions(n, seed):
        t, p = _clip(c)
        fn = _save_spectrum(out, t, p, rng)
        rec[fn] = {"temperature_C": round(t, 2), "pH": round(p, 2)}
        print(f"  {fn}  (T={t:.1f}, pH={p:.2f})  selectivity~{selectivity(t, p):.2f}")
    (out / "conditions.json").write_text(json.dumps(rec, indent=2))
    _write_experiment_txt(out, "spectra", _rebuild_manifest(out))


def cmd_spectra_run(out: Path, params, seed):
    rng = np.random.default_rng(seed)
    rec = json.loads((out / "conditions.json").read_text()) if (out / "conditions.json").is_file() else {}
    for p in params:
        t, ph = _clip(p)
        fn = _save_spectrum(out, t, ph, rng)
        rec[fn] = {"temperature_C": round(t, 2), "pH": round(ph, 2)}
        print(f"  {fn}  (T={t:.1f}, pH={ph:.2f})  selectivity~{selectivity(t, ph):.2f}")
    (out / "conditions.json").write_text(json.dumps(rec, indent=2))
    _write_experiment_txt(out, "spectra", _rebuild_manifest(out))


# ---------------------------------------------------------------- tidy mode
def _append_results(out: Path, params, seed):
    import csv
    rng = np.random.default_rng(seed)
    path = out / "results.csv"
    new = not path.is_file()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["temperature_C", "pH", "product_area", "byproduct_area"])
        for p in params:
            t, ph = _clip(p)
            a_p = max(product_area(t, ph) * (1 + rng.normal(0, 0.03)), 0.0)
            a_b = max(byproduct_area(t, ph) * (1 + rng.normal(0, 0.03)), 1e-3)
            w.writerow([f"{t:.2f}", f"{ph:.2f}", f"{a_p:.4f}", f"{a_b:.4f}"])
            print(f"  row T={t:.1f}, pH={ph:.2f}  product={a_p:.3f} byproduct={a_b:.3f} "
                  f"selectivity~{a_p / a_b:.2f}")


def cmd_tidy_init(out: Path, n, seed):
    _append_results(out, _init_conditions(n, seed), seed)
    _write_experiment_txt(out, "tidy", None)


def cmd_tidy_run(out: Path, params, seed):
    _append_results(out, params, seed)
    _write_experiment_txt(out, "tidy", None)


# ---------------------------------------------------------------- reset
def cmd_reset(out: Path, mode, n, seed):
    """Wipe a campaign's artifacts in ``out`` and re-seed the initial batch.

    A plain re-``init`` does NOT reset cleanly: spectra mode leaves orphan
    run-emitted ``spectrum_*.csv`` files behind, and tidy mode APPENDS to the
    existing ``results.csv`` (doubling the seed rows). This clears only the
    known campaign files, then re-seeds deterministically (same ``seed`` ->
    identical initial batch). It does NOT touch the downstream SciLink session.
    """
    patterns = (
        ["spectrum_*.csv", "conditions.json"] if mode == "spectra"
        else ["results.csv"]
    ) + ["experiment.txt"]
    removed = 0
    for pat in patterns:
        for f in out.glob(pat):
            f.unlink()
            removed += 1
    print(f"  cleared {removed} campaign file(s) in {out}/")
    (cmd_spectra_init if mode == "spectra" else cmd_tidy_init)(out, n, seed)


# ---------------------------------------------------------------- cli
def _parse_params(s):
    p = json.loads(s)
    return [p] if isinstance(p, dict) else p


def main():
    ap = argparse.ArgumentParser(description="Reaction-screening simulator for SciLink BO")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="seed an initial batch (far from the optimum)")
    pi.add_argument("--mode", choices=["spectra", "tidy"], required=True)
    pi.add_argument("--out", required=True)
    pi.add_argument("--n", type=int, default=8)
    pi.add_argument("--seed", type=int, default=42)

    pr = sub.add_parser("run", help="emit measurement(s) at BO-suggested params")
    pr.add_argument("--mode", choices=["spectra", "tidy"], required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--params", required=True, help='JSON dict or list of {temperature_C, pH}')
    pr.add_argument("--seed", type=int, default=0)

    prst = sub.add_parser("reset", help="wipe campaign artifacts and re-seed the initial batch")
    prst.add_argument("--mode", choices=["spectra", "tidy"], required=True)
    prst.add_argument("--out", required=True)
    prst.add_argument("--n", type=int, default=8)
    prst.add_argument("--seed", type=int, default=42)

    sub.add_parser("truth", help="print the ground-truth selectivity optimum")

    args = ap.parse_args()
    if args.cmd == "truth":
        t, p, s = ground_truth_optimum()
        print(f"Ground-truth optimum: temperature_C={t:.1f}, pH={p:.2f}, selectivity={s:.2f}")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "init":
        (cmd_spectra_init if args.mode == "spectra" else cmd_tidy_init)(out, args.n, args.seed)
        print(f"\nInitialized {args.mode} demo in {out}/  (metadata -> experiment.txt)")
        t, p, s = ground_truth_optimum()
        print(f"Ground-truth optimum (hidden): T={t:.1f} C, pH={p:.2f}, selectivity={s:.2f}")
    elif args.cmd == "reset":
        cmd_reset(out, args.mode, args.n, args.seed)
        print(f"\nReset {args.mode} demo in {out}/ to initial state ({args.n} measurement(s)).")
        print("Note: the SciLink meta session (BO history/uploads) is separate — "
              "start a fresh session or clear its uploads to fully restart the campaign.")
    else:
        params = _parse_params(args.params)
        (cmd_spectra_run if args.mode == "spectra" else cmd_tidy_run)(out, params, args.seed)
        print(f"\nGenerated {len(params)} new {args.mode} measurement(s) in {out}/")


if __name__ == "__main__":
    main()
