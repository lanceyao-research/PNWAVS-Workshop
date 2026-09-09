# Running a Closed-Loop Experiment Campaign with SciLink

This shows how you'd run a **self-driving optimization campaign on your own
instrument**: you do a few experiments, SciLink analyzes them and tells you the
*next* conditions most worth trying, you run that experiment, and the loop
repeats — converging on the best conditions in far fewer experiments than a grid
search.

> **What's real vs. simulated here.** Everything you'd actually do — record your
> data + conditions, upload them, read SciLink's analysis, get the next
> recommendation — is exactly as it would be in your lab. The **only** stand-in
> is the measurement itself: where you'd walk to your instrument and run the next
> experiment, here a tiny script (`simulate.py`) plays the instrument and returns
> a realistic data file. Swap that one step for your real instrument and the
> workflow is identical.

**The example campaign.** A reaction makes a desired **product** and an unwanted
**byproduct**; you can set two knobs — **temperature** and **pH** — and you want
to maximize **selectivity** (product ÷ byproduct). Nobody tells SciLink where the
best conditions are; it discovers them from your data. You start with a handful
of poor early experiments and watch it climb.

## You describe your experiment in plain language

Conditions live in **`experiment.txt`** — written the way a scientist actually
would: the objective, the knob ranges your instrument allows, and which file was
run at which conditions. There's **no rigid format to learn** — SciLink reads
your description. (For the "raw spectra" case it turns that into per-file metadata
for you automatically.)

## Two ways scientists start — pick the one that matches your data

### A. You collect raw measurements (e.g. spectra) — `spectra_demo/`
SciLink does the analysis *and* the optimization.
1. Upload the **`spectra_demo/`** folder (the spectra + `experiment.txt`) to SciLink.
2. Ask it, in plain language:
   > "Analyze these UV-Vis spectra, extract the product and byproduct peak areas
   > vs temperature and pH, then run Bayesian optimization to maximize selectivity
   > and tell me the next experiment to run."
3. SciLink fits the peaks, tabulates the results, and recommends the next
   (temperature, pH).
4. **Run that experiment.** In your lab you'd measure it; here:
   ```bash
   python simulate.py run --mode spectra --out ./spectra_demo \
       --params '{"temperature_C": 60, "pH": 7.5}'      # the values SciLink recommended
   ```
5. Upload the new spectrum, ask SciLink to update and recommend again. Repeat —
   the recommendations march toward the best conditions.

### B. You already have a results table — `tidy_demo/`
If you've already reduced your data to numbers (one row per experiment), SciLink
skips analysis and goes straight to recommending the next experiment.
1. Upload **`tidy_demo/results.csv`** (+ `experiment.txt`).
2. Ask: "Maximize selectivity = product_area / byproduct_area; what should I run next?"
3. Run the recommended experiment and add the new row; repeat:
   ```bash
   python simulate.py run --mode tidy --out ./tidy_demo \
       --params '{"temperature_C": 60, "pH": 7.5}'
   ```

## What "good" looks like

The campaign starts with selectivity below ~1 and should climb toward the best
region over several rounds. For the presenter only, the true optimum is:
```bash
python simulate.py truth      # ~ temperature 55 C, pH 6.6, selectivity 3.6
```
A scientist running this for real wouldn't know that number in advance — that's
the point: SciLink finds it from the experiments.

## Hands-free version (optional)

To watch the whole loop run automatically and get a convergence plot
(`convergence.png`):
```bash
export AWS_BEARER_TOKEN_BEDROCK=...   AWS_REGION_NAME=us-east-1   UNSAFE_EXECUTION_OK=true
python run_loop.py --mode tidy    --iters 5 --base ./_loop_tidy
python run_loop.py --mode spectra --iters 4 --base ./_loop_spectra
```

## Start over

```bash
rm -rf spectra_demo tidy_demo
python simulate.py init --mode spectra --out ./spectra_demo --n 9
python simulate.py init --mode tidy    --out ./tidy_demo    --n 9
```

`simulate.py` needs only `numpy`. The hands-free `run_loop.py` additionally needs
`scilink` + Bedrock credentials. See `README.md` for the technical reference.
