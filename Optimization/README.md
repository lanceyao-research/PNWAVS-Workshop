# Reaction-Screening Simulator for SciLink's BO Loop

A richer cousin of `MRSTutorial_MT01/BOAgent`. Instead of a single peak, the
simulated "reaction" produces **two competing species** — a desired **product**
and an unwanted **byproduct** — each giving a resolvable UV-Vis peak. You
optimize the **selectivity** = `product_peak_area / byproduct_peak_area` by
tuning two knobs: **temperature_C** and **pH**.

Two emit modes match SciLink's two routing paths:

| Mode | Emits | Meta routes to | What runs |
|---|---|---|---|
| `spectra` | `wavelength_nm,absorbance` CSVs (a measurement **curve**) | **analysis** → planning | multi-peak curve fit → feature table → scalarizer → BO |
| `tidy` | rows in `results.csv` (`temperature_C,pH,product_area,byproduct_area`) | **planning** directly | scalarizer (computes the ratio) → BO |

Conditions/metadata are written to a human-readable **`experiment.txt`** (not
sidecar JSON): objective, parameter bounds, and — in `spectra` mode — a per-file
conditions manifest. Stating the bounds in `experiment.txt` lets the BO path run
without pausing to ask for them.

## Ground truth (intentionally hard)

- The product forms along a **curved ridge** in (T, pH) — the optimal pH rises
  with temperature (a banana-shaped valley) with a single best temperature.
- The byproduct grows at high pH (hydrolysis) and high T (decomposition).
- So the selectivity optimum is **interior and off-center** (~ T≈55 °C, pH≈6.6),
  and `init` deliberately seeds the campaign in a **far low-T / low-pH corner**
  so you can watch BO travel across the space and converge.

`python simulate.py truth` prints the optimum (for your reference — it is **not**
written into `experiment.txt`).

## Quick start (interactive, e.g. via the UI)

```bash
# Spectra path (full analyze -> feature-table -> BO loop)
python simulate.py init --mode spectra --out ./spectra_demo --n 9
#   -> upload ./spectra_demo/ (the spectra + experiment.txt) to SciLink.
#      Objective is already in experiment.txt: "maximize selectivity".
#   -> SciLink analyzes (fits product+byproduct peaks), builds a feature table,
#      scalarizes selectivity, runs BO, and recommends the next (temperature_C, pH).

# Generate the recommended measurement and upload it; repeat.
python simulate.py run --mode spectra --out ./spectra_demo \
    --params '{"temperature_C": 60, "pH": 7.5}'
```

```bash
# Tidy path (straight to planning's scalarizer + BO)
python simulate.py init --mode tidy --out ./tidy_demo --n 8
#   -> upload ./tidy_demo/results.csv (+ experiment.txt) to SciLink (Plan mode).
python simulate.py run --mode tidy --out ./tidy_demo \
    --params '[{"temperature_C": 60, "pH": 7.5}, {"temperature_C": 65, "pH": 8}]'
```

## Watch it converge (headless driver)

`run_loop.py` drives the meta agent in a closed loop against the simulator and
saves a `convergence.png` (recommendation path over the true selectivity surface
+ a best-so-far curve):

```bash
export AWS_BEARER_TOKEN_BEDROCK=...   AWS_REGION_NAME=us-east-1   UNSAFE_EXECUTION_OK=true
python run_loop.py --mode tidy    --iters 5 --base ./_loop_tidy
python run_loop.py --mode spectra --iters 2 --base ./_loop_spectra
```

## Commands

| Command | What it does |
|---|---|
| `init --mode {spectra,tidy} --out DIR [--n N]` | seed an initial batch in the far corner + write `experiment.txt` |
| `run --mode {spectra,tidy} --out DIR --params '{...}'` | emit measurement(s) at the BO-suggested point(s); `--params` takes a dict or a list |
| `reset --mode {spectra,tidy} --out DIR [--n N]` | wipe campaign artifacts in DIR and re-seed the initial batch (deterministic). Use the same `--n`/`--seed` as `init` to restore the exact starting set. Does **not** touch the SciLink meta session — start a fresh session (or clear its uploads/`bo_artifacts`) to fully restart the campaign. |
| `truth` | print the (hidden) ground-truth selectivity optimum |

Both knobs are clipped to the instrument ranges (`temperature_C` 5–100, `pH`
1–14). Spectra carry mild solvatochromic peak shifts, a sloping baseline,
run-to-run height variability, and detector noise so the fit and the GP residuals
are realistic.
