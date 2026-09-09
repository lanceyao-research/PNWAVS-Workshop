"""Drive the SciLink meta agent in a closed BO loop against the simulator, and
record how the recommendations converge toward the (hidden) selectivity optimum.

  python run_loop.py --mode tidy    --iters 5 --base ./_loop_tidy
  python run_loop.py --mode spectra --iters 2 --base ./_loop_spectra

Each iteration: feed the current data to the meta (which routes spectra->analysis
->planning and tidy->planning), parse the BO-recommended (temperature_C, pH) from
a machine-readable NEXT line, then synthesize that measurement so it's present for
the next iteration. At the end, print the trajectory and save convergence.png.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simulate as SIM  # noqa: E402

os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
MODEL = "bedrock/us.anthropic.claude-opus-4-8"
NEXT_RE = re.compile(r"NEXT\s+temperature_C\s*=\s*([0-9.]+)\s+pH\s*=\s*([0-9.]+)", re.I)
TAIL = ("Finish your reply with EXACTLY one machine-readable line:\n"
        "NEXT temperature_C=<value> pH=<value>")


def banner(m): print(f"\n{'#'*72}\n{m}\n{'#'*72}", flush=True)


def prompt_for(mode, data, it):
    if mode == "spectra":
        if it == 0:
            return (
                f"I'm running a UV-Vis reaction-screening campaign. The spectra "
                f"(columns wavelength_nm, absorbance) are in: {data}\n"
                f"The file {data/'experiment.txt'} states the objective, the allowed "
                "parameter ranges, and the per-file conditions (temperature_C, pH).\n"
                "Please: (1) analyze the spectra by curve fitting — resolve the product "
                "peak near 540 nm and the byproduct peak near 430 nm and extract each "
                "peak's AREA; (2) build a feature table of those areas vs temperature_C "
                "and pH; (3) run Bayesian optimization to maximize "
                "selectivity = product_area / byproduct_area over the stated ranges, and "
                "recommend the single next (temperature_C, pH) to measure. Proceed fully "
                "autonomously end to end; do not pause to ask. " + TAIL)
        return (
            f"I ran the recommended experiment — a new spectrum was added to {data} and "
            f"{data/'experiment.txt'} was updated with its conditions. Re-analyze the "
            "updated set (reuse the same fitting recipe), update the Bayesian "
            "optimization campaign, and recommend the next single (temperature_C, pH). "
            "Proceed autonomously. " + TAIL)
    # tidy
    if it == 0:
        return (
            f"I have a reaction-screening results table at {data/'results.csv'} "
            "(columns temperature_C, pH, product_area, byproduct_area). The file "
            f"{data/'experiment.txt'} states the objective and the allowed parameter "
            "ranges. The objective is to maximize "
            "selectivity = product_area / byproduct_area. Process this data and run "
            "Bayesian optimization to recommend the single next (temperature_C, pH) over "
            "the stated ranges. Proceed fully autonomously; do not pause. " + TAIL)
    return (
        f"I ran it — a new row was appended to {data/'results.csv'}. Update the Bayesian "
        "optimization campaign with the new point and recommend the next single "
        "(temperature_C, pH). Proceed autonomously. " + TAIL)


def parse_next(reply):
    m = NEXT_RE.search(reply or "")
    if m:
        return float(m.group(1)), float(m.group(2))
    # fallback: last "temperature_C = X" / "pH = Y" mentioned
    t = re.findall(r"temperature_C[^0-9\-]{0,6}([0-9]{1,3}(?:\.[0-9]+)?)", reply or "", re.I)
    p = re.findall(r"pH[^0-9\-]{0,6}([0-9]{1,2}(?:\.[0-9]+)?)", reply or "", re.I)
    if t and p:
        return float(t[-1]), float(p[-1])
    return None, None


def save_plot(base, traj, mode):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    Topt, pHopt, _ = SIM.ground_truth_optimum()
    ts = np.linspace(*SIM.BOUNDS["temperature_C"], 160)
    ps = np.linspace(*SIM.BOUNDS["pH"], 160)
    TT, PP = np.meshgrid(ts, ps)
    S = SIM.selectivity(TT, PP)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    c = ax1.contourf(TT, PP, S, levels=20, cmap="viridis")
    fig.colorbar(c, ax=ax1, label="true selectivity")
    xs = [t for _, t, _, _ in traj]; ys = [p for _, _, p, _ in traj]
    ax1.plot(xs, ys, "-o", color="white", mfc="red", mec="white", ms=8, label="BO recommendations")
    for it, t, p, _ in traj:
        ax1.annotate(str(it + 1), (t, p), color="white", fontsize=8, ha="center", va="center")
    ax1.plot(Topt, pHopt, "*", color="gold", ms=22, mec="k", label=f"optimum ({Topt:.0f},{pHopt:.1f})")
    ax1.set_xlabel("temperature_C"); ax1.set_ylabel("pH")
    ax1.set_title(f"{mode}: BO recommendation path"); ax1.legend(loc="upper left", fontsize=8)
    best = np.maximum.accumulate([s for *_, s in traj])
    ax2.plot(range(1, len(traj) + 1), [s for *_, s in traj], "o--", label="recommended-point selectivity")
    ax2.plot(range(1, len(traj) + 1), best, "-", lw=2, label="best so far")
    ax2.axhline(SIM.ground_truth_optimum()[2], color="gold", ls=":", label="true optimum")
    ax2.set_xlabel("iteration"); ax2.set_ylabel("true selectivity at recommended point")
    ax2.set_title("convergence"); ax2.legend(fontsize=8)
    fig.tight_layout()
    out = Path(base) / "convergence.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["spectra", "tidy"], required=True)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--n_init", type=int, default=8)
    ap.add_argument("--base", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path(args.base); base.mkdir(parents=True, exist_ok=True)
    data = base / "data"; data.mkdir(exist_ok=True)
    session = base / "meta_session"

    banner(f"INIT {args.mode} batch (far from optimum)")
    if args.mode == "spectra":
        SIM.cmd_spectra_init(data, args.n_init, args.seed)
    else:
        SIM.cmd_tidy_init(data, args.n_init, args.seed)
    Topt, pHopt, Sopt = SIM.ground_truth_optimum()
    print(f"(hidden) optimum T={Topt:.1f} pH={pHopt:.2f} selectivity={Sopt:.2f}", flush=True)

    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode
    agent = MetaOrchestratorAgent(base_dir=str(session), model_name=MODEL,
                                  meta_mode=MetaMode.AUTONOMOUS)

    traj = []
    for it in range(args.iters):
        banner(f"ITERATION {it + 1}/{args.iters}  ({args.mode})")
        reply = agent.chat(prompt_for(args.mode, data, it))
        print(reply, flush=True)
        T, pH = parse_next(reply)
        if T is None:
            print("!! could not parse a NEXT recommendation — stopping", flush=True)
            break
        T = float(np.clip(T, *SIM.BOUNDS["temperature_C"]))
        pH = float(np.clip(pH, *SIM.BOUNDS["pH"]))
        sel = SIM.selectivity(T, pH)
        traj.append((it, T, pH, sel))
        print(f">>> iter {it + 1}: BO recommends T={T:.1f}, pH={pH:.2f}  "
              f"(true selectivity there = {sel:.2f}; optimum = {Sopt:.2f})", flush=True)
        # synthesize the measurement so it's present for the next iteration
        p = [{"temperature_C": T, "pH": pH}]
        if args.mode == "spectra":
            SIM.cmd_spectra_run(data, p, args.seed + it + 1)
        else:
            SIM.cmd_tidy_run(data, p, args.seed + it + 1)

    banner("TRAJECTORY")
    print(f"{'iter':>4} {'temp_C':>8} {'pH':>6} {'true_sel':>9}")
    for it, T, pH, sel in traj:
        print(f"{it + 1:>4} {T:>8.1f} {pH:>6.2f} {sel:>9.2f}")
    if traj:
        best = max(s for *_, s in traj)
        print(f"\nbest true selectivity reached: {best:.2f} / optimum {Sopt:.2f} "
              f"({100 * best / Sopt:.0f}% of optimum)")
        save_plot(base, traj, args.mode)


if __name__ == "__main__":
    main()
