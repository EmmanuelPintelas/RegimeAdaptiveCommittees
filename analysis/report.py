"""Tables and figures for the paper, from results/main/*.json.

    python -m analysis.report

Writes paper/generated/tab_*.tex (complete table environments), paper/figures/*.pdf
and results/main/summary.md.  No number in these tables is typed by hand.  Oracle
references are cross-fitted (analysis.crossfit), and differences on the test split
carry paired cluster-bootstrap intervals (analysis.bootstrap): a cell
``+0.073 +- 0.012`` is the point estimate with half the width of the 95% percentile
interval, starred when the interval excludes zero.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .pools import CODE

RESULTS = os.path.join(CODE, "results", "main")
OUTPUT = os.path.join(CODE, "results")
GEN = os.path.join(OUTPUT, "tables")
FIG = os.path.join(OUTPUT, "figures")

BENCH = [("camelyon17", "Camelyon17", 2), ("kather", "Kather", 9),
         ("iwildcam", "iWildCam", 182), ("rxrx1", "RxRx1", 1139)]
TUNED = ("camelyon17", "iwildcam", "rxrx1")
NAMES = {k: l for k, l, _ in BENCH}
BACKBONE = {"efficientnet_b0": "EffNet-B0", "convnext_atto": "ConvNeXt-A",
            "regnety_004": "RegNetY"}
AGG = "avg"
METRIC_SHORT = {"gm": "GM", "macro_f1": "F1"}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load():
    res = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if not isinstance(r, dict) or "sweep" not in r or r.get("mode", "build") == "refine":
            continue
        res.setdefault(r["benchmark"], []).append(r)
    order = {"seed_1": 0, "convnext": 1, "mixed": 2}
    for b in res:
        res[b].sort(key=lambda r: next((v for k, v in order.items() if k in r["pool"]), 9))
    return res


def load_json(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


CF = None
BS = None


def cf(r):
    return (CF or {}).get(r["pool"])


_BS_CACHE = {}


def boot(r, name):
    """Bootstrap replicates of an estimator for one pool, or None."""
    idx = (BS or {}).get(r["pool"])
    if not idx or name not in idx["keys"]:
        return None
    if r["pool"] not in _BS_CACHE:
        _BS_CACHE[r["pool"]] = np.load(os.path.join(RESULTS, "bootstrap", f"{r['pool']}.npz"))
    return _BS_CACHE[r["pool"]][idx["keys"][name]]


def point(r, name):
    idx = (BS or {}).get(r["pool"])
    return None if not idx else idx["points"].get(name)


def diff(rs, a, b):
    """Mean over pools of score(a) - score(b), with its paired 95% interval."""
    pts, reps = [], []
    for r in rs:
        pa, pb, ra, rb = point(r, a), point(r, b), boot(r, a), boot(r, b)
        if pa is None or pb is None or ra is None or rb is None:
            return None
        pts.append(pa - pb)
        reps.append(ra - rb)
    reps = np.mean(reps, axis=0)
    return float(np.mean(pts)), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def fmt_ci(d, digits=3):
    if d is None:
        return "--"
    p, lo, hi = d
    star = r"^{\ast}" if lo > 0 or hi < 0 else ""
    return rf"${p:+.{digits}f}{{\scriptstyle\pm{(hi - lo) / 2:.{digits}f}}}{star}$"


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def backbone(r):
    if len(r["archs"]) == 1:
        return BACKBONE.get(next(iter(r["archs"])), next(iter(r["archs"])))
    return "mixed"


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def signed(x):
    return "--" if x is None else f"{x:+.3f}"


def ks(v):
    return "/".join(str(int(k)) for k in v) if isinstance(v, (list, tuple)) else str(v)


def rows_by_benchmark(res, make_row, ncols, benches=None):
    lines = []
    for key, label, K in BENCH:
        if benches is not None and key not in benches:
            continue
        pools = res.get(key, [])
        for i, r in enumerate(pools):
            head = [label if i == 0 else "", str(K) if i == 0 else "", backbone(r)]
            lines.append(" & ".join(head + make_row(r)) + r" \\")
        lines.append(r"\addlinespace")
    if lines and lines[-1] == r"\addlinespace":
        lines.pop()
    return lines


def table(name, caption, colspec, header, body, size=r"\footnotesize", tabcolsep=None,
          place="t"):
    sep = rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}" if tabcolsep else ""
    tex = "\n".join([
        rf"\begin{{table}}[{place}]", r"\centering", size,
        rf"\caption{{{caption}}}", rf"\label{{tab:{name}}}",
        sep + r"\begin{adjustbox}{max width=\linewidth}",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""])
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, f"tab_{name}.tex"), "w", encoding="utf-8") as f:
        f.write(tex)


def curves(r, agg, split):
    return r["sweep"][agg][split]["curves"]


def best_k(curve):
    return int(np.argmax(curve)) + 1


def size_stats(r, agg=AGG):
    c = curves(r, agg, r["test"])
    met = r["metric"]
    g = np.asarray(c["greedy"][met])
    kb = best_k(g)
    return {"full": c["random"][met][-1], "k": kb, "best": float(g[kb - 1]),
            "gap": float(g[kb - 1] - c["topk"][met][kb - 1]),
            "random1": c["random"][met][0], "conf_best": max(c["confidence"][met])}


# --------------------------------------------------------------------------- #
# tables: pools and geometry
# --------------------------------------------------------------------------- #
def tab_pools(res):
    body = []
    for key, label, K in BENCH:
        rs = res.get(key, [])
        if not rs:
            continue
        r = rs[0]
        t, v = r["test"], r["tuning"]
        n_v = r["splits"][v]["n"] if v else None
        body.append(" & ".join([label, str(K), METRIC_SHORT[r["metric"]], str(len(rs)),
                                str(r["M"]), f"{n_v:,}".replace(",", r"\,") if n_v else "--",
                                f"{r['splits'][t]['n']:,}".replace(",", r"\,")]) + r" \\")
    table("pools", "Benchmarks and pools: one pool per backbone plan (EfficientNet-B0, "
          "ConvNeXt-atto, and a mix with RegNetY-400MF). Metric: geometric mean of recalls "
          "(GM) for $K<11$, macro-F1 otherwise.", "lrlrrrr",
          r"benchmark & $K$ & metric & pools & $M$ & $n$ tuning & $n$ test", body,
          place="h")


def tab_geometry(res):
    gaps = []

    def row(r):
        v = r["tuning"] or r["test"]
        c = r["splits"][v]["collision"]
        floor = 1.0 / max(r["K"] - 1, 1)
        g = r.get("geometry") or {}
        mu_s = r["splits"][v].get("mu_split")
        if mu_s is not None:
            gaps.append(c["frac_mu_negative"] - mu_s)
        return [f3(c["mean_member_error"]), f3(c["q"]), f3(c["q3"]),
                f"{c['q'] / floor:.0f}", f"{1 / c['q']:.1f}", f3(mu_s),
                f3(g.get("error_alignment_wrong")), f3(g.get("ambiguity_share"))]
    body = rows_by_benchmark(res, row, 11)
    gap = max(gaps) if gaps else float("nan")
    table("geometry", r"Error geometry, tuning split (test split for Kather). $\coll$, "
          r"$\coll^{(3)}$: pooled two- and three-fold collision probability; $\times$floor "
          r"$=\coll(K-1)$; $K_{\mathrm{eff}}=1/\coll$. $\mu<0$: share of points where the "
          r"strongest wrong label out-polls the true one, chosen on half of the members and "
          rf"counted on the other (on the same votes: higher by at most {gap:.3f}). Err.\ "
          r"align.: mean cosine of members' error vectors on wrong-label coordinates. Amb.\ "
          r"share: ambiguity term of \eqref{eq:ambiguity} over mean member Brier risk.",
          "lrlcccccccc",
          r"benchmark & $K$ & backbone & member err. & $\coll$ & $\coll^{(3)}$ & $\times$floor & "
          r"$K_{\mathrm{eff}}$ & $\mu<0$ & err.\ align. & amb.\ share", body, tabcolsep="3pt")


# --------------------------------------------------------------------------- #
# tables: oracle references
# --------------------------------------------------------------------------- #
def tab_size(res):
    def row(r):
        s = size_stats(r)
        c = cf(r)
        if c is None:
            return [f3(s["full"]), f3(s["random1"]), str(s["k"]), f3(s["best"])] + ["--"] * 4
        a = c["avg"]
        srch = c["search"]["sizes"][str(c["search"]["k_ref"])]
        gain = diff([r], "cross-fitted greedy reference", "full pool (halves)")
        gap = diff([r], "cross-fitted greedy reference", "cross-fitted top-k at its size")
        return [f3(s["full"]), f3(s["random1"]), str(s["k"]), f3(s["best"]), ks(a["k"]),
                fmt_ci(gain), signed(srch["best"] - srch["greedy"]), fmt_ci(gap)]
    body = rows_by_benchmark(res, row, 11)
    table("size", r"Test split, probability averaging. Random $k{=}1$: a random member. "
          r"Greedy reference (test labels): in-sample size and score; cross-fitted sizes per "
          r"direction and gain over the full pool on the same halves (paired 95\% interval). "
          r"Search: the best of three stronger searches minus greedy, at the in-sample size. "
          r"Last column: cross-fitted greedy minus top-$k$ at greedy's size.", "lrlcccccccc",
          r"& & & & & \multicolumn{2}{c}{in-sample} & \multicolumn{2}{c}{cross-fitted} & & \\ "
          r"\cmidrule(lr){6-7}\cmidrule(lr){8-9} benchmark & $K$ & backbone & full pool & "
          r"random $k{=}1$ & $k$ & score & $k$ & gain & search & greedy $-$ top-$k$", body,
          tabcolsep="3pt")


def tab_gain(res):
    ratios = []

    def row(r):
        g = r["selection_gain"]
        c = cf(r)
        ratios.append(g.get("bound_ties_as_errors", np.nan) / g["bound"] if g["bound"] else np.nan)
        if c is None:
            return [f3(g["full_err"]), f3(g["headroom"]), f3(g["minority_right_mass"])] + ["--"] * 6
        gc = c["gain"]
        return [f3(g["full_err"]), f3(g["headroom"]), f3(g["minority_right_mass"]),
                f3(gc["best_member_share"]), str(g["k"]), signed(g["realized_gain"]),
                signed(gc["gain"]), f3(g["bound"]), f3(c["certificate"]["B"])]
    body = rows_by_benchmark(res, row, 12)
    table("gain", r"What selection can gain: test split, plurality vote (ties by confidence), accuracy. Headroom: "
          r"points the full pool gets wrong and some member gets right; minority-right: those "
          r"where the strongest wrong label out-polls the truth in expectation. Best member: "
          r"cross-fitted share of minority-right points one member covers. Gain: greedy "
          r"reference (${\le}40$ members, test labels) minus the full pool. Bound: "
          r"\eqref{eq:gain} with $\nu^{\ge}_k$ ($\nu^{>}_k$ divides it by up to "
          rf"{1 / np.nanmin(ratios) if ratios else 2:.0f}); $B(k)$: "
          r"Proposition~\ref{prop:certificate}.",
          "lrlccccccccc",
          r"& & & & & & & & \multicolumn{2}{c}{gain} & & \\ \cmidrule(lr){9-10} "
          r"benchmark & $K$ & backbone & full err. & headroom & minority-right & best member & "
          r"$k$ & in-sample & cross-fitted & bound & $B(k)$", body, tabcolsep="3pt")


def span(vals, fmt="{:.3f}"):
    """'a--b' over pools, or 'a' when they agree at the printed precision."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return lo if lo == hi else f"{lo}--{hi}"


def tab_soft(res):
    body = []
    for key, label, K in BENCH:
        rs = [r for r in res.get(key, []) if cf(r)]
        if not rs:
            continue
        cells = []
        for agg in ("vote", "wvote", "avg"):
            cells += [span([cf(r)[agg]["score"] for r in rs]),
                      span([cf(r)[agg]["full"] for r in rs])]
        body.append(" & ".join([label, str(K)] + cells) + r" \\")
    table("soft", "Cross-fitted greedy reference and the full pool on the same halves of the "
          "test split, under the three aggregations (range over pools).", "lr" + "cc" * 3,
          r"& & \multicolumn{2}{c}{plurality vote} & \multicolumn{2}{c}{confidence-weighted} "
          r"& \multicolumn{2}{c}{average} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}"
          r"\cmidrule(lr){7-8} benchmark & $K$ & greedy & full & greedy & full & greedy & full",
          body, tabcolsep="4pt")


def tab_marginal(res):
    def row(r):
        met = r["metric"]
        tc = r["p7_test_curves"]
        pr = r["proposers"]
        k, kz = pr["p7_stop"], pr["p7_z_stop"]
        full = curves(r, "vote", r["test"])["random"][met][-1]
        c = cf(r)
        out = [f3(full), str(k), f3(tc["P7 collision-aware"][k - 1]), str(kz),
               f3(tc["P7 pool-agreement"][kz - 1])]
        if c is None:
            return out + ["--"] * 4
        return out + [ks(c["p7_oracle"]["k"]), signed(c["p7_oracle"]["score"] - c["vote"]["full"]),
                      ks(c["vote"]["k"]), signed(c["vote"]["score"] - c["vote"]["full"])]
    body = rows_by_benchmark(res, row, 12)
    table("marginal", r"Collision-aware greedy (P7), test split, plurality vote (ties by confidence). Confidence, pool "
          r"agreement: label-free plug-ins, each path at its own size $\hat k$. Oracle: test "
          r"correctness as plug-ins, built on one half and scored on the other (size per "
          r"direction; difference from the full pool on the same halves). Greedy: the "
          r"cross-fitted reference.",
          "lrlccccccccc",
          r"& & & & \multicolumn{2}{c}{P7, confidence} & \multicolumn{2}{c}{P7, pool agr.} & "
          r"\multicolumn{2}{c}{P7, oracle} & \multicolumn{2}{c}{greedy} \\ "
          r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12} "
          r"benchmark & $K$ & backbone & full & $\hat k$ & score & $\hat k$ & score & $k$ & "
          r"$-$ full & $k$ & $-$ full", body, tabcolsep="2.5pt")


def tab_ancestry(res):
    d = load_json("ancestry.json")
    if not d:
        return
    body = []
    for r in d["benchmarks"]:
        s, i = r["siblings"], r["independent"]
        sr, si = r["size_refined"], r["size_independent"]
        cr = sr.get("crossfit") or []
        ci = si.get("crossfit") or []
        body.append(" & ".join([
            NAMES[r["benchmark"]], str(r["K"]),
            f3(s["H"]["error"]), f3(i["H"]["error"]),
            f3(s["H"]["collision"]), f3(i["H"]["collision"]),
            f3(s["E"]["collision"]), f3(i["E"]["collision"]), f3(r["theta"]),
            f"{sr['k']} ({sr['distinct_parents']})", str(si["k"]),
            "/".join(f"{c['k']} ({c['distinct_parents']})" for c in cr) or "--",
            "/".join(str(c["k"]) for c in ci) or "--"]) + r" \\")
    table("ancestry", r"Shared ancestry, test split. Siblings (children of one parent) against "
          r"independent members of the parent's pool, on the parent's mistakes and elsewhere: "
          r"error rate and collision probability, pooled over pairs, points and parents. "
          r"$\hat\theta$: copy rate (Proposition~\ref{prop:ancestry}). Greedy-reference size "
          r"under the vote, in-sample and cross-fitted per direction; in brackets, distinct "
          r"parents.", "lr" + "c" * 11,
          r"& & \multicolumn{2}{c}{error, parent wrong} & \multicolumn{2}{c}{$\coll$, parent "
          r"wrong} & \multicolumn{2}{c}{$\coll$, elsewhere} & & \multicolumn{2}{c}{size, "
          r"in-sample} & \multicolumn{2}{c}{size, cross-fitted} \\ \cmidrule(lr){3-4}"
          r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){10-11}\cmidrule(lr){12-13} "
          r"benchmark & $K$ & sib. & ind. & sib. & ind. & sib. & ind. & $\hat\theta$ & refined & "
          r"ind. & refined & ind.", body, tabcolsep="3pt")


# --------------------------------------------------------------------------- #
# tables: selection with a tuning split
# --------------------------------------------------------------------------- #
def short_source(src):
    if src in ("held-out greedy", "held-out top-k"):
        return {"held-out greedy": "greedy", "held-out top-k": "top-$k$"}[src]
    if src == "full pool":
        return "full"
    return src.split(" ")[0]


def tab_method(res):
    def row(r):
        m = r["method"][AGG]
        met = r["metric"]
        mn, un = m["main"], m["uniform_prior"]
        full = m["full_pool_test"]
        cons = mn.get("conservative")
        mv = r["method"].get("vote") or {}
        vote = (signed(mv["main"]["test"][met] - mv["full_pool_test"][met]) if mv else "--")
        return [short_source(mn["source"]), str(mn["k"]), f3(mn["eps"]),
                f3(mn["test"][met]), fmt_ci(diff([r], "decider", "full pool")),
                signed(mn["test"]["acc"] - full["acc"]),
                signed(cons["test"][met] - full[met]) if cons else "--",
                f"{short_source(un['source'])} {un['k']}", signed(un["test"][met] - full[met]),
                f3(full[met]), vote]
    body = rows_by_benchmark(res, row, 14, benches=TUNED)
    table("method", r"Algorithm~\ref{alg:select} with a tuning split; test split, probability "
          r"averaging. Chosen: a proposed path, or a search family refitted on all of $V$ at "
          r"size $k$; $\varepsilon$: its deviation. $-$ full: difference from the full pool "
          r"(paired 95\% interval), and in accuracy. Conservative: the held-out committee "
          r"instead of the refit. Uniform prior: its choice and difference. Vote: the decider "
          r"minus the full pool under the plurality vote.",
          "lrl" + "c" * 11,
          r"& & & \multicolumn{5}{c}{decider} & & \multicolumn{2}{c}{uniform prior} & & \\ "
          r"\cmidrule(lr){4-8}\cmidrule(lr){10-11} benchmark & $K$ & backbone & chosen & $k$ & "
          r"$\varepsilon$ & score & $-$ full & $-$ full (acc.) & conservative & choice & $-$ full "
          r"& full pool & vote", body, tabcolsep="2.5pt")


def tab_gamma(res):
    def sgn(v):
        return "--" if v is None else f"{v:+.3f}"

    def row(r):
        m = r["method"][AGG]
        ti = m["insample_greedy"]["transfer"]
        tr = m["main"]["transfer"]
        return [sgn(ti["a_V"]), sgn(ti["a_T"]), sgn(ti["regret"]),
                sgn(tr["a_V"]), sgn(tr["a_T"]), f3(tr["gamma"]), f3(tr["dis_T"]),
                sgn(tr["regret"])]
    body = rows_by_benchmark(res, row, 11, benches=TUNED)
    table("gamma", r"Transfer against the full pool, probability averaging, plain error "
          r"(Proposition~\ref{prop:transfer}). $a_V$, $a_{\mathrm T}$: disagreement advantage of "
          r"the full pool over the chosen committee on the tuning and test splits (negative: the "
          r"committee wins where the two disagree); $\gamma=|a_{\mathrm T}-a_V|$; "
          r"$\mathrm{dis}_{\mathrm T}$: label-free test disagreement; regret: test error minus "
          r"the full pool's.", "lrl" + "c" * 8,
          r"& & & \multicolumn{3}{c}{in-sample greedy} & \multicolumn{5}{c}{decider} \\ "
          r"\cmidrule(lr){4-6}\cmidrule(lr){7-11} "
          r"benchmark & $K$ & backbone & $a_V$ & $a_{\mathrm T}$ & regret & $a_V$ & "
          r"$a_{\mathrm T}$ & $\gamma$ & $\mathrm{dis}_{\mathrm T}$ & regret", body,
          tabcolsep="3.5pt")


BASELINE_ROWS = [
    ("decider", r"decider (Algorithm~\ref{alg:select})"),
    ("decider (uniform prior)", "decider, uniform prior"),
    ("decider (conservative)", "decider, conservative deployment"),
    ("ablation: full pool vs held-out greedy", "full pool against held-out greedy"),
    ("ablation: menu, held-out greedy only", "menu with held-out greedy only"),
    ("ablation: menu, held-out top-k only", "menu with held-out top-$k$ only"),
    ("best member on V", "best member on $V$"),
    ("top-k on V", "top-$k$ on $V$, $k$ chosen on $V$"),
    ("greedy on V", "greedy on $V$, $k$ chosen on $V$"),
    ("Caruana (bagged)", "ensemble selection (Caruana)"),
    ("log-odds weighted vote", "log-odds weighted vote"),
    ("linear-pool stacking", "linear-pool stacking"),
    ("proposed paths by balanced accuracy", "proposed paths, by balanced accuracy"),
    ("proposed paths by NLL", "proposed paths, by NLL"),
    ("proposed paths by BRIER", "proposed paths, by Brier score"),
]


def bench_pools(res, b):
    return [r for r in res.get(b, []) if r["method"].get(AGG)]


def tab_baselines(res):
    benches = [b for b in TUNED if bench_pools(res, b)]
    body = []
    for key, label in BASELINE_ROWS:
        cells = []
        for b in benches:
            rs = bench_pools(res, b)
            d = diff(rs, key, "full pool")
            worst = min((point(r, key) - point(r, "full pool")) for r in rs) \
                if d is not None else None
            cells += [fmt_ci(d), signed(worst)]
        body.append(label + " & " + " & ".join(cells) + r" \\")
        if key in ("decider (conservative)", "ablation: menu, held-out top-k only",
                   "linear-pool stacking"):
            body.append(r"\addlinespace")
    header = (" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{NAMES[b]}}}" for b in benches)
              + r" \\ " + "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                                  for i in range(len(benches)))
              + " method & " + " & ".join("mean & worst" for _ in benches))
    table("baselines", r"Rules that choose on the tuning split $V$: test score minus the full "
          r"pool's, probability averaging (log-odds: a weighted vote), mean over each "
          r"benchmark's pools with paired 95\% interval, and worst pool. Proposed paths: the "
          r"menu prefix best on $V$ by balanced accuracy, NLL or Brier score.",
          "l" + "c" * (2 * len(benches)), header, body, tabcolsep="3pt")


BUDGET_ROWS = [
    ("decider", r"decider (Algorithm~\ref{alg:select})"),
    ("random", "random committee"),
    ("source-validation top-k", "source-validation top-$k$"),
    ("confidence top-k", "confidence top-$k$ (P1)"),
    ("P7 collision-aware", "collision-aware path (P7)"),
    ("P11", "signal consensus, mean (P11)"),
    ("P12", "signal consensus, geometric (P12)"),
    ("top-k on V", "top-$k$ on $V$"),
    ("greedy on V", "greedy on $V$"),
    ("Caruana (capped)", "ensemble selection (Caruana)"),
]


def budget_value(r, b, key):
    """Score at budget b minus the full pool's; P11/P12 averaged over spectral versions."""
    row = r["baselines"]["budget"][str(b)]
    full = r["method"][AGG]["full_pool_test"][r["metric"]]
    if key in ("P11", "P12"):
        vals = [v[r["metric"]] for n, v in row.items() if n.startswith(key) and isinstance(v, dict)]
        return float(np.mean(vals)) - full
    return row[key][r["metric"]] - full


def budget_halfwidth(rs, b, key):
    """Half-width of the paired 95% interval of a budget row against the full pool, or None."""
    reps = []
    for r in rs:
        idx = (BS or {}).get(r["pool"])
        if not idx:
            return None
        names = ([n for n in idx["keys"] if n.startswith(f"budget {b}: {key}")]
                 if key in ("P11", "P12") else [f"budget {b}: {key}"])
        if not names or any(n not in idx["keys"] for n in names):
            return None
        reps.append(np.mean([boot(r, n) for n in names], axis=0) - boot(r, "full pool"))
    lo, hi = np.percentile(np.mean(reps, axis=0), [2.5, 97.5])
    return float(hi - lo) / 2


def tab_budget(res):
    from .selection import BUDGETS
    benches = [b for b in TUNED if bench_pools(res, b)]
    body = []
    widths = {b: [] for b in benches}
    for key, label in BUDGET_ROWS:
        cells = []
        for b in benches:
            rs = bench_pools(res, b)
            for k in BUDGETS:
                cells.append(signed(float(np.mean([budget_value(r, k, key) for r in rs]))))
                if key != "random":
                    w = budget_halfwidth(rs, k, key)
                    if w is not None:
                        widths[b].append(w)
        body.append(label + " & " + " & ".join(cells) + r" \\")
        if key in ("random", "P12"):
            body.append(r"\addlinespace")
    spans = [f"{max(widths[b]):.3f} on {NAMES[b]}" for b in benches if widths[b]]
    ci = ""
    if spans:
        ci = (" Paired 95\\% intervals against the full pool (every row but random) have "
              "half-widths of at most " + ", ".join(spans[:-1]) + " and " + spans[-1] + ".")
    nb = len(BUDGETS)
    header = (" & " + " & ".join(rf"\multicolumn{{{nb}}}{{c}}{{{NAMES[b]}}}" for b in benches)
              + r" \\ " + "".join(rf"\cmidrule(lr){{{2 + nb * i}-{1 + nb * (i + 1)}}}"
                                  for i in range(len(benches)))
              + " $k\\le k_{\\max}$ & " + " & ".join(" & ".join(str(k) for k in BUDGETS)
                                                    for _ in benches))
    table("budget", r"A budget of $k_{\max}$ members, full pool excluded: test score minus the "
          r"full pool's, probability averaging, mean over each benchmark's pools. Rules on $V$ "
          r"choose sizes up to $k_{\max}$; label-free paths are cut at $k_{\max}$; random: "
          r"mean of twenty draws." + ci,
          "l" + "c" * (nb * len(benches)), header, body, tabcolsep="3pt")


def tab_menus(res):
    names = None
    for b in TUNED:
        for r in bench_pools(res, b):
            names = [n for n in r["method"][AGG]["ablations"]
                     if not n.startswith(("menu,", "full pool"))]
            break
        if names:
            break
    if not names:
        return
    benches = [b for b in TUNED if bench_pools(res, b)]
    body = []
    for n in names:
        vals_all = []
        cells = []
        for b in benches:
            vs = []
            for r in bench_pools(res, b):
                m = r["method"][AGG]
                vs.append(m["ablations"][n]["test"][r["metric"]] - m["full_pool_test"][r["metric"]])
            vals_all += vs
            cells.append(signed(float(np.mean(vs))))
        label = n.replace("-", "--").replace(" no search", ", no search")
        if n == "P7,P11-P12":
            label = r"\textbf{" + label + " (method)}"
        body.append(label + " & " + " & ".join(cells) + f" & {signed(float(np.mean(vals_all)))}"
                    f" & {signed(min(vals_all))}" + r" \\")
    table("menus", r"The decider under the menus of the ablation, the held-out search families "
          r"and the regime prior the same in all: test score minus the full pool's, probability "
          r"averaging, mean over the three pools of each benchmark, over all nine pools, and the "
          r"worst pool. P7,P11--P12 is the menu of the method; the menus without search are "
          r"decided on their proposed paths alone.", "l" + "c" * (len(benches) + 2),
          "menu & " + " & ".join(NAMES[b] for b in benches) + " & all & worst", body,
          tabcolsep="4pt")


def tab_qhat(res):
    body = []
    for key, label, K in BENCH:
        rs = res.get(key, [])
        if not rs:
            continue
        lf = [r["label_free"] for r in rs]
        ece = [r["splits"][r["test"]].get("calibration", {}).get("ece_mean") for r in rs]
        body.append(" & ".join([label, str(K), span([d["e_z"] for d in lf]),
                                span([d["q_labels"] for d in lf]), span([d["q_free"] for d in lf]),
                                span([d.get("q_corrected") for d in lf]),
                                span([d["sampling_bound"] for d in lf], "{:.2f}"),
                                span(ece, "{:.2f}")]) + r" \\")
    table("qhat", r"Label-free collision estimate, test split, range over pools. $e_z$: error of "
          r"the pool plurality; $\bar\coll$: with test labels; $\tilde\coll$: against the "
          r"plurality; corrected: with the excluded masses (Proposition~\ref{prop:qhat}) "
          r"estimated on $V$; sampling: bound \eqref{eq:qhat-sample}, $\eta=0.05$. ECE: "
          r"members' mean calibration error (15 bins).",
          "lrcccccc",
          r"benchmark & $K$ & $e_z$ & $\bar\coll$ & $\tilde\coll$ & corrected & "
          r"sampling & ECE", body, tabcolsep="3.5pt")


def tab_proposers(res):
    pools = [r for b in TUNED for r in bench_pools(res, b)]
    if not pools:
        return
    names = []
    for r in pools:
        for n in r["method"][AGG]["per_proposer"]:
            if n not in names:
                names.append(n)
    # one row per proposer family: on how many pools one of its paths is the best proposed
    # path, and its paths' difference from the full pool, worst and best over pools
    fam_names = {"P1": "confidence", "P2": "spectral exponent", "P3": "pool agreement",
                 "P4": "pseudo-label complementarity", "P5": "diversity-regularized conf.",
                 "P6": "source validation", "P7": "collision-aware greedy",
                 "P8": "weight-noise stability", "P9": "consensus of proposers, mean",
                 "P10": "consensus of proposers, geometric", "P11": "consensus of signals, mean",
                 "P12": "consensus of signals, geometric"}
    fams = list(dict.fromkeys(n.split()[0] for n in names))
    best_count = {f: 0 for f in fams}
    diffs = {f: [] for f in fams}
    for r in pools:
        m = r["method"][AGG]
        full = m["full_pool_test"][r["metric"]]
        pp = m["per_proposer"]
        top = max(pp, key=lambda n: pp[n]["test_at_kV"])
        best_count[top.split()[0]] += 1
        for f in fams:
            vals = [pp[n]["test_at_kV"] - full for n in pp if n.split()[0] == f]
            if vals:
                diffs[f].append((min(vals), max(vals)))
    body = []
    for f in fams:
        lo = min(d[0] for d in diffs[f])
        hi = max(d[1] for d in diffs[f])
        n_paths = sum(1 for n in names if n.split()[0] == f)
        body.append(f"({f}) {fam_names.get(f, f)} & {n_paths} & {len(diffs[f])} & "
                    f"{best_count[f]} & {signed(lo)} & {signed(hi)}" + r" \\")
    table("proposers", "Each proposer family at the size best on the tuning split (balanced "
          "accuracy), scored on the test split, probability averaging, over the nine pools: "
          "its paths (directions, scales or weights), the pools it applies to, the pools on "
          "which one of its paths is the best proposed path, and the difference of its paths "
          "from the full pool, worst and best.", "lccccc",
          r"proposer & paths & pools & best on & worst $-$ full & best $-$ full", body,
          tabcolsep="4pt")


SIGNAL_NAMES = {"P1": "confidence (P1)", "P2": "spectral exponent (P2)",
                "P3": "agreement with the pool (P3)", "P6": "source validation (P6)",
                "P8": "weight-noise stability (P8)"}


def _signal_values(subsets, tag):
    return {frozenset() if k == "none" else frozenset(k.split("+")): row[tag]
            for k, row in subsets.items()}


def _shapley(v, signals):
    from itertools import combinations
    from math import factorial
    n = len(signals)
    phi = {}
    for j in signals:
        others = [s for s in signals if s != j]
        tot = 0.0
        for s in range(n):
            for T in combinations(others, s):
                w = factorial(s) * factorial(n - s - 1) / factorial(n)
                tot += w * (v[frozenset(T) | {j}] - v[frozenset(T)])
        phi[j] = tot
    return phi


def tab_signals(res):
    d = load_json("signal_ablation.json")
    if not d:
        return
    signals = d["signals"]
    benches = [b for b in TUNED if any(p["benchmark"] == b for p in d["pools"])]
    cols = {}
    for b in benches:
        pools = [p for p in d["pools"] if p["benchmark"] == b]
        for tag in ("with", "without"):
            vs = [_signal_values(p["subsets"], tag) for p in pools]
            phis = [_shapley(v, signals) for v in vs]
            cols[(b, tag)] = {**{s: np.mean([ph[s] for ph in phis]) for s in signals},
                              "all": np.mean([v[frozenset(signals)] for v in vs]),
                              "none": np.mean([v[frozenset()] for v in vs])}
    keys = [(b, t) for b in benches for t in ("with", "without")]

    def cells(key):
        return " & ".join(f"{100 * cols[k][key]:+.2f}" for k in keys)
    body = [f"{SIGNAL_NAMES[s]} & {cells(s)} " + r"\\" for s in signals]
    body += [r"\midrule", f"all five signals & {cells('all')} " + r"\\",
             f"no signal & {cells('none')} " + r"\\"]
    header = (" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{NAMES[b]}}}" for b in benches)
              + r" \\ " + "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                                  for i in range(len(benches)))
              + " signal & " + " & ".join("method & alone" for _ in benches))
    table("signals", r"Shapley value of each consensus signal (its gain averaged over all removal "
          r"orders), test score minus the full pool's in points ($\times100$), mean over each "
          r"benchmark's pools. Method: the decider with P7 and held-out search; alone: the "
          r"consensus paths only. A column sums to \emph{all five} minus \emph{no signal}.",
          "l" + "c" * len(keys), header, body, tabcolsep="4pt")


def tab_purenoise(res):
    d = load_json("pure_noise.json")
    if not d:
        return
    rows = d["rows"]
    ks_ = [k for k in rows[0]["by_k"] if k in ("1", "5", "10", "30")]
    body = [f"{k} & " + " & ".join(f"{r['by_k'][k]['mean']:.3f} / {r['by_k'][k]['bound']:.3f}"
                                   for r in rows) + r" \\" for k in ks_]
    body += [r"\midrule",
             r"lower bound (ii), $k=1$ & " + " & ".join(f3(r["lower_bound_k1"]) for r in rows)
             + r" \\"]
    table("purenoise", rf"Greedy subset search with pure-noise members: $M={d['M']}$ members, "
          rf"each right on each tuning point independently with probability {d['accuracy']:g}, "
          rf"two classes, plurality vote with ties as errors. Cells: tuning accuracy of the "
          rf"greedy prefix of size $k$ minus its true accuracy, mean over {d['reps']} draws, / "
          rf"the deviation of Proposition~\ref{{prop:greedy-cost}} with $k_{{\max}}=k$ at "
          rf"$\eta={d['eta']:g}$.", "l" + "c" * len(rows),
          r"$k$ & " + " & ".join(f"$n={r['n']:,}$".replace(",", r"{,}") for r in rows), body,
          tabcolsep="5pt")


def tab_headline(res):
    """One row per benchmark: regime, greedy-reference size and gain, decider, budget."""
    body = []
    for key, label, K in BENCH:
        rs = res.get(key, [])
        if not rs:
            continue
        met = METRIC_SHORT[rs[0]["metric"]]
        v = rs[0]["tuning"] or rs[0]["test"]
        q = [r["splits"][r["tuning"] or r["test"]]["collision"]["q"] for r in rs]
        cfs = [cf(r) for r in rs]
        sizes = [k for c in cfs if c for k in c["avg"]["k"]]
        gain = diff(rs, "cross-fitted greedy reference", "full pool (halves)")
        if key in TUNED and bench_pools(res, key):
            dec = diff(rs, "decider", "full pool")
            b10 = [budget_value(r, 10, "decider") for r in rs]
            best_base = max(np.mean([budget_value(r, 10, k) for r in rs])
                            for k, _ in BUDGET_ROWS[1:])
            budget = f"{np.mean(b10):+.3f} / {best_base:+.3f}"
        else:
            dec, budget = None, "--"
        body.append(" & ".join([label, str(K), met, f"{min(q):.2f}--{max(q):.2f}",
                                f"{min(sizes)}--{max(sizes)}" if sizes else "--",
                                fmt_ci(gain), fmt_ci(dec), budget]) + r" \\")
    table("headline", r"Headline numbers, test split, probability averaging, mean over each "
          r"benchmark's pools with paired 95\% intervals. $\coll$: range over pools. Greedy "
          r"reference (test labels): cross-fitted size range and gain. Decider, and budget "
          r"$k\le10$ (decider / best other rule of Table~\ref{tab:budget}): minus the full pool.",
          "lrlccccc",
          r"benchmark & $K$ & metric & $\coll$ & size & gain & decider & budget $k\le10$",
          body, tabcolsep="3pt")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
# Every figure is drawn at the text width (6.0 in) and included at \linewidth, so
# the font sizes below are the printed sizes.
TEXT_WIDTH = 6.0
SIZE_TICKS = [1, 2, 5, 10, 20, 50, 100]
METRIC_NAME = {"gm": "geometric mean of recalls", "macro_f1": "macro-F1"}


def _size_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(SIZE_TICKS)
    ax.set_xticklabels([str(t) for t in SIZE_TICKS])
    ax.minorticks_off()


def fig_ksweep(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    benches = [(k, l) for k, l, _ in BENCH if k in res]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 3.9), constrained_layout=True)
    # random is the label-free baseline (neutral); the three ranked rules take the
    # first three categorical slots in order, with line style as secondary encoding.
    # Label-free curves are differences from the full pool on the whole test split;
    # the oracle curves are cross-fitted, differences from the full pool on the halves.
    for ax, (key, label) in zip(axes.ravel(), benches):
        pools = res[key]
        met = pools[0]["metric"]
        k = np.arange(1, pools[0]["M"] + 1)
        ax.axhline(0, color=INK, lw=0.9, label="full pool", zorder=2)
        series = []
        for rule, col, ls, lab in (("random", MUTED, "-", "random"),
                                   ("confidence", SERIES[1], "-", "confidence")):
            c = np.array([np.asarray(curves(r, AGG, r["test"])[rule][met])
                          - curves(r, AGG, r["test"])["random"][met][-1] for r in pools])
            series.append((c, col, ls, lab, 1.6))
        cfs = [cf(r) for r in pools]
        if all(cfs):
            for kk, col, ls, lab in (("topk_curve", SERIES[2], "--", "top-$k$ (cross-fitted)"),
                                     ("greedy_curve", SERIES[0], ":",
                                      "greedy (cross-fitted)")):
                c = np.array([np.asarray(x["avg"][kk]) - x["avg"]["full"] for x in cfs])
                series.append((c, col, ls, lab, 1.8))
        ins = np.array([np.asarray(curves(r, AGG, r["test"])["greedy"][met])
                        - curves(r, AGG, r["test"])["random"][met][-1] for r in pools])
        series.append((ins, SERIES[0], (0, (1, 3)), "greedy (in-sample)", 0.9))
        for c, col, ls, lab, lw in series:
            ax.plot(k, c.mean(0), color=col, ls=ls, lw=lw, label=lab, zorder=3)
            if lw > 1:
                ax.fill_between(k, c.min(0), c.max(0), color=col, alpha=0.12, lw=0)
        _size_axis(ax)
        ax.set_title(f"{label} ($K={pools[0]['K']}$)", fontsize=9)
        if ax in axes[:, 0]:
            ax.set_ylabel(f"{METRIC_NAME[met]}\nminus full pool", fontsize=8.5)
        _chrome(ax)
    for ax in axes[1]:
        ax.set_xlabel("committee size $k$", fontsize=8.5)
    fig.legend(handles=axes[0][0].get_legend_handles_labels()[0], loc="outside upper center",
               ncol=3, fontsize=8, frameon=False, labelcolor=INK2)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "ksweep.pdf"))
    plt.close(fig)


def fig_predicted(res):
    """Random committees: measured plurality error against the two predictions,
    one panel per benchmark, mean over its pools (bands: range over pools)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    benches = [(k, l) for k, l, _ in BENCH if k in res]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 3.8), constrained_layout=True)
    for ax, (key, label) in zip(axes.ravel(), benches):
        pools = res[key]
        ds = [r["splits"][r["tuning"] or r["test"]] for r in pools]
        ks = np.asarray(ds[0]["random_pred_q"]["ks"])
        assert all(list(d["random_pred_q"]["ks"]) == list(ks) for d in ds)
        meas = np.array([np.asarray(d["random_measured"]["err_mean"])[ks - 1] for d in ds])
        pq = np.array([d["random_pred_q"]["err"] for d in ds])
        pk = np.array([d["random_pred_K"]["err"] for d in ds])
        Lq = sorted({int(d["random_pred_q"]["L"]) for d in ds})
        LK = int(ds[0]["random_pred_K"]["L"])
        lines = [(meas, INK, "-", 1.6, "measured"),
                 (pq, SERIES[0], "--", 1.6, "predicted from $q$")]
        if Lq != [LK]:
            lines.append((pk, SERIES[1], ":", 1.8, "predicted from $K$"))
        for v, col, ls, lw, lab in lines:
            ax.plot(ks, v.mean(0), color=col, ls=ls, lw=lw, label=lab, zorder=3)
            ax.fill_between(ks, v.min(0), v.max(0), color=col, alpha=0.12, lw=0)
        lq = f"{Lq[0]}" if len(Lq) == 1 else f"{Lq[0]}$–${Lq[-1]}"
        title = (f"{label}: $L={lq}$ from $q$ and $K$" if Lq == [LK]
                 else f"{label}: $L={lq}$ from $q$, ${LK}$ from $K$")
        ax.set_title(title, fontsize=9)
        _size_axis(ax)
        _chrome(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("plurality error", fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("committee size $k$", fontsize=8.5)
    handles = axes[0][1].get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=8, frameon=False,
               labelcolor=INK2)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "predicted_curve.pdf"))
    plt.close(fig)


INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")            # categorical slots 1-3, validated
BLUE_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")
SHAPES = {"camelyon17": "o", "kather": "s", "iwildcam": "^", "rxrx1": "D"}
REGIME_MIN_POINTS = 20        # minority-right and correctable points a regime-map cell needs


def _chrome(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def fig_regime_map():
    """Regime map: collision against concentration, and concentration against gain."""
    path = os.path.join(RESULTS, "regime_map.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    cells = [c for c in regime_cells(cells)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.lines import Line2D
    cmap = LinearSegmentedColormap.from_list("size", BLUE_RAMP)
    norm = LogNorm(vmin=1, vmax=40)
    names = dict((k, l) for k, l, _ in BENCH)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8), constrained_layout=True)
    panels = [("q", "cf_share", "collision probability $q$",
               "best member's share\nof minority-right points", True),
              ("cf_share", "cf_recovered",
               "best member's share of minority-right points",
               "share of correctable\nmistakes recovered", False)]
    for ax, (xk, yk, xl, yl, xlog) in zip(axes, panels):
        for b, mk in SHAPES.items():
            for full in (False, True):
                sel = [c for c in cells if c["benchmark"] == b and (c["cell"] == "full") == full]
                if not sel:
                    continue
                ax.scatter([c[xk] for c in sel], [c[yk] for c in sel], c=[c["k_cf"] for c in sel],
                           cmap=cmap, norm=norm, marker=mk, s=58 if full else 22,
                           edgecolors=INK if full else "white", linewidths=1.1 if full else 0.6,
                           zorder=3 if full else 2)
        if xlog:
            ax.set_xscale("log")
            ax.set_xticks([0.1, 0.2, 0.5, 1.0])
            ax.set_xticklabels(["0.1", "0.2", "0.5", "1"])
            ax.minorticks_off()
        ax.set_xlabel(xl, fontsize=8.5)
        ax.set_ylabel(yl, fontsize=8.5)
        _chrome(ax)
    handles = [Line2D([], [], ls="", marker=mk, ms=7, mfc=MUTED, mec=INK2, mew=0.6,
                      label=names[b])
               for b, mk in SHAPES.items() if any(c["benchmark"] == b for c in cells)]
    handles += [Line2D([], [], ls="", marker="o", ms=8.5, mfc="white", mec=INK, mew=1.1,
                       label="full problem"),
                Line2D([], [], ls="", marker="o", ms=6.5, mfc=MUTED, mec=INK2, mew=0.4,
                       label="class subset")]
    axes[1].legend(handles=handles, fontsize=7.5, loc="upper left", frameon=False,
                   labelcolor=INK2)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.9, pad=0.01)
    cb.set_ticks([1, 2, 5, 10, 20, 40])
    cb.set_ticklabels(["1", "2", "5", "10", "20", "40"])
    cb.set_label("greedy-reference size $k$", color=INK2, fontsize=8)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_edgecolor(AXIS)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "regime_map.pdf"))
    plt.close(fig)


def _kilo(n):
    if n < 1000:
        return str(n)
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return f"{n / 1000:.1f}k" if n < 10000 else f"{n / 1000:.0f}k"


def fig_tuning_size():
    """Decider and in-sample tuning greedy against the size of the tuning split."""
    d = load_json("tuning_size.json")
    if not d:
        return
    pools = d["pools"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    benches = [b for b, _, _ in BENCH if any(p["benchmark"] == b for p in pools)]
    fig, axes = plt.subplots(1, len(benches), figsize=(TEXT_WIDTH, 2.7), squeeze=False,
                             constrained_layout=True)
    for col, b in enumerate(benches):
        ps = [p for p in pools if p["benchmark"] == b]
        ns = sorted({row["n"] for row in ps[0]["rows"]})
        ax = axes[0][col]
        ax.axhline(0, color=MUTED, lw=1.0, ls=":", zorder=1)
        for key, colr, ls, mk, lab in (("decider", SERIES[0], "-", "o", "decider"),
                                       ("tuning_greedy", SERIES[1], "--", "s",
                                        "in-sample greedy")):
            mean, lo, hi = [], [], []
            for n in ns:
                vals = [row[key] - p["full_pool"] for p in ps for row in p["rows"]
                        if row["n"] == n]
                mean.append(np.mean(vals))
                lo.append(np.percentile(vals, 10))
                hi.append(np.percentile(vals, 90))
            ax.plot(ns, mean, color=colr, ls=ls, lw=1.8, marker=mk, ms=4.5, label=lab, zorder=3)
            ax.fill_between(ns, lo, hi, color=colr, alpha=0.13, lw=0)
        ax.set_title(f"{NAMES[b]} ({ {'gm': 'geometric mean', 'macro_f1': 'macro-F1'}[ps[0]['metric']] })",
                     fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels(["" if i == len(ns) - 2 and ns[-1] / n < 2 else _kilo(n)
                            for i, n in enumerate(ns)], fontsize=7.5)
        ax.minorticks_off()
        ax.set_xlim(ns[0] * 0.75, ns[-1] * 1.35)
        ax.set_xlabel("size of the tuning split $n$", fontsize=8.5)
        _chrome(ax)
    axes[0][0].set_ylabel("test score minus full pool", fontsize=8.5)
    fig.legend(handles=axes[0][0].get_legend_handles_labels()[0], loc="outside upper center",
               ncol=2, fontsize=8, frameon=False, labelcolor=INK2)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "tuning_size.pdf"))
    plt.close(fig)


def regime_cells(cells, convention=None):
    """Cells with enough minority-right and correctable points and cross-fitted terms."""
    out = []
    for c in cells:
        src = c if convention is None else c.get(convention)
        if src is None or c["n"] * c["minority_right_mass"] < REGIME_MIN_POINTS \
                or c["n"] * c["headroom"] < REGIME_MIN_POINTS:
            continue
        if src.get("cf_share") is None or src.get("cf_recovered") is None:
            continue
        d = dict(c)
        d.update({k: src[k] for k in ("q", "cf_share", "cf_recovered", "cf_k")})
        d["k_cf"] = float(np.exp(np.mean(np.log(src["cf_k"]))))
        out.append(d)
    return out


def regime_stats():
    """Spearman correlations of the regime map: across all cells (descriptive), within
    benchmarks with a cluster bootstrap over pools, per pool, and under the alternative
    subset convention."""
    from scipy.stats import spearmanr
    d = load_json("regime_map.json")
    if not d:
        return None
    rng = np.random.default_rng(0)
    out = {}
    for conv in (None, "alt"):
        cells = regime_cells(d["cells"], conv)
        if conv == "alt":
            # the alternative convention exists for class subsets only; add the full problems
            cells += [c for c in regime_cells(d["cells"]) if c["cell"] == "full"]
        pairs = {"share-recovered": ("cf_share", "cf_recovered"), "q-share": ("q", "cf_share"),
                 "share-k": ("cf_share", "k_cf"), "q-k": ("q", "k_cf"), "K-k": ("K", "k_cf")}
        res = {"n_cells": len(cells),
               "all": {n: float(spearmanr([c[a] for c in cells], [c[b] for c in cells])[0])
                       for n, (a, b) in pairs.items()}}
        for bench in ("kather", "iwildcam", "rxrx1"):
            bc = [c for c in cells if c["benchmark"] == bench]
            pools = sorted({c["pool"] for c in bc})
            if len(bc) < 5:
                continue
            row = {}
            for n, (a, b) in pairs.items():
                est = float(spearmanr([c[a] for c in bc], [c[b] for c in bc])[0])
                reps = []
                for _ in range(1000):
                    pick = rng.choice(pools, len(pools), replace=True)
                    sample = [c for p in pick for c in bc if c["pool"] == p]
                    if len({c[a] for c in sample}) > 1 and len({c[b] for c in sample}) > 1:
                        reps.append(spearmanr([c[a] for c in sample], [c[b] for c in sample])[0])
                per_pool = []
                for p in pools:
                    pc = [c for c in bc if c["pool"] == p]
                    if len(pc) >= 4:
                        per_pool.append(float(spearmanr([c[a] for c in pc], [c[b] for c in pc])[0]))
                row[n] = {"est": est, "lo": float(np.nanpercentile(reps, 2.5)),
                          "hi": float(np.nanpercentile(reps, 97.5)),
                          "per_pool": per_pool}
            res[bench] = row
        out["main" if conv is None else "alt"] = res
    return out


def fig_signals():
    d = load_json("signal_ablation.json")
    if not d:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    signals = d["signals"]
    benches = [b for b in TUNED if any(p["benchmark"] == b for p in d["pools"])]
    fig, axes = plt.subplots(1, len(benches), figsize=(TEXT_WIDTH, 2.5),
                             squeeze=False, constrained_layout=True)
    for ax, b in zip(axes[0], benches):
        pools = [p for p in d["pools"] if p["benchmark"] == b]
        ax.axhline(0, color=MUTED, lw=1.0, ls=":", zorder=1)
        for tag, colr, ls, mk, lab in (("with", SERIES[0], "-", "o", "method"),
                                       ("without", SERIES[1], "--", "s", "consensus alone")):
            vs = [_signal_values(p["subsets"], tag) for p in pools]
            mean_v = {T: np.mean([v[T] for v in vs]) for T in vs[0]}
            kk = np.arange(len(signals) + 1)
            by = [[val for T, val in mean_v.items() if len(T) == k] for k in kk]
            ax.plot(kk, [np.mean(x) for x in by], color=colr, ls=ls, lw=1.8, marker=mk, ms=4.5,
                    label=lab, zorder=3)
            ax.fill_between(kk, [min(x) for x in by], [max(x) for x in by], color=colr,
                            alpha=0.13, lw=0)
        ax.set_xticks(range(len(signals) + 1))
        ax.set_xlabel("signals kept", fontsize=8.5)
        ax.set_title(f"{NAMES[b]} ({ {'gm': 'geometric mean', 'macro_f1': 'macro-F1'}[pools[0]['metric']] })",
                     fontsize=9)
        _chrome(ax)
    axes[0][0].set_ylabel("test score minus full pool", fontsize=8.5)
    fig.legend(handles=axes[0][0].get_legend_handles_labels()[0], loc="outside upper center",
               ncol=2, fontsize=8, frameon=False, labelcolor=INK2)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "signal_ablation.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# summary for writing
# --------------------------------------------------------------------------- #
def summary(res):
    L = ["# Results (results/main)", ""]
    for key, label, K in BENCH:
        for r in res.get(key, []):
            met = r["metric"]
            L.append(f"## {label} / {backbone(r)} ({r['pool']})")
            s = size_stats(r)
            c = cf(r)
            L.append(f"- greedy ref in-sample k={s['k']} {s['best']:.4f} (full {s['full']:.4f}); "
                     f"conf best {s['conf_best']:.4f}")
            if c:
                a = c["avg"]
                g = diff([r], "cross-fitted greedy reference", "full pool (halves)")
                gap = diff([r], "cross-fitted greedy reference", "cross-fitted top-k at its size")
                L.append(f"- cross-fitted greedy k={a['k']} score {a['score']:.4f} (full halves "
                         f"{a['full']:.4f}); gain {g}; greedy-topk {gap}")
                sr = c["search"]["sizes"]
                L.append("- search: " + "; ".join(
                    f"k={k}: greedy {v['greedy']:.4f} best {v['best']:.4f} (rand "
                    f"{v['random_start']}, swap {v['swap']:.4f}, back {v['backward']})"
                    for k, v in sr.items()))
                gc = c["gain"]
                L.append(f"- gain cf {gc['gain']:+.4f} (k {gc['k']}), best member share cf "
                         f"{gc['best_member_share']}, certificate B({c['certificate']['k']}) = "
                         f"{c['certificate']['B']:.4f}; B by k {c['certificate']['B_by_k']}; "
                         f"mean D on H {c['certificate']['mean_D_on_H']}")
                L.append(f"- soft cf: " + "; ".join(f"{agg} {c[agg]['score']:.4f} k {c[agg]['k']} "
                                                     f"full {c[agg]['full']:.4f}"
                                                     for agg in ("vote", "wvote", "avg")))
                L.append(f"- P7 oracle cf: k {c['p7_oracle']['k']} {c['p7_oracle']['score']:.4f}")
            g = r["selection_gain"]
            L.append(f"- gain in-sample k={g['k']} {g['realized_gain']:+.4f} bound {g['bound']:.4f} "
                     f"(ties as errors {g.get('bound_ties_as_errors', float('nan')):.4f}) headroom "
                     f"{g['headroom']:.4f} minority-right {g['minority_right_mass']:.4f} best member "
                     f"{g['best_member_cover_minority']:.4f}")
            lf = r["label_free"]
            cal = r["splits"][r["test"]].get("calibration", {})
            L.append(f"- q bar {lf['q_labels']:.4f} q tilde {lf['q_free']:.4f} corrected "
                     f"{lf.get('q_corrected')} ece {cal.get('ece_mean')}")
            v = r["tuning"] or r["test"]
            L.append(f"- mu<0 split {r['splits'][v].get('mu_split')} plain "
                     f"{r['splits'][v]['collision']['frac_mu_negative']:.4f}; err align wrong "
                     f"{(r.get('geometry') or {}).get('error_alignment_wrong')}")
            m = r["method"].get(AGG)
            if m:
                mn = m["main"]
                L.append(f"- decider {mn['source']} k={mn['k']} eps {mn['eps']:.4f} -> "
                         f"{mn['test'][met]:.4f} (full {m['full_pool_test'][met]:.4f}); "
                         f"diff {diff([r], 'decider', 'full pool')}; uniform prior "
                         f"{m['uniform_prior']['source']} k={m['uniform_prior']['k']} "
                         f"{m['uniform_prior']['test'][met]:.4f}; conservative "
                         f"{(mn.get('conservative') or {}).get('test', {}).get(met)}")
                L.append(f"- transfer decider {mn['transfer']}; in-sample greedy "
                         f"{m['insample_greedy']['transfer']} test {m['insample_greedy']['test'][met]:.4f}")
                for n, d in m["ablations"].items():
                    L.append(f"  - ablation {n}: {d['source']} k={d['k']} {d['test'][met]:.4f}")
                for b, d in m["budgets"].items():
                    L.append(f"  - budget {b}: {d['source']} k={d['k']} {d['test'][met]:.4f}")
                B = r["baselines"]
                for n in ("best member on V", "top-k on V", "greedy on V", "Caruana (bagged)",
                          "log-odds weighted vote", "linear-pool stacking",
                          "proposed paths by balanced accuracy", "proposed paths by NLL",
                          "proposed paths by BRIER"):
                    L.append(f"  - baseline {n}: k={B[n].get('k')} {B[n].get('source', '')} "
                             f"{B[n]['test'][met]:.4f} diff {diff([r], n, 'full pool')}")
                if "inductive" in r:
                    ind = r["inductive"]
                    L.append(f"- inductive: {ind['inductive']['source']} k={ind['inductive']['k']} "
                             f"{ind['inductive']['test'][met]:.4f}; transductive "
                             f"{ind['transductive']['test'][met]:.4f}; full "
                             f"{ind['full_pool'][met]:.4f} (eval half, {ind['halves']})")
            mv = r["method"].get("vote")
            if mv:
                L.append(f"- vote decider {mv['main']['source']} k={mv['main']['k']} "
                         f"{mv['main']['test'][met]:.4f} (full {mv['full_pool_test'][met]:.4f})")
            L.append("")
    rs = regime_stats()
    if rs:
        L.append("## regime map")
        L.append(json.dumps(rs, indent=1))
    with open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    global CF, BS
    res = load()
    CF = load_json("crossfit.json")
    BS = load_json("bootstrap.json")
    for t in (tab_pools, tab_geometry, tab_size, tab_gain, tab_soft, tab_marginal, tab_ancestry,
              tab_method, tab_gamma, tab_baselines, tab_budget, tab_menus, tab_qhat,
              tab_proposers, tab_signals, tab_purenoise, tab_headline):
        try:
            t(res)
        except Exception as e:                    # report what is missing, keep going
            print(f"{t.__name__}: {type(e).__name__}: {e}")
    for f in (lambda: fig_ksweep(res), lambda: fig_predicted(res), fig_regime_map,
              fig_tuning_size, fig_signals):
        try:
            f()
        except Exception as e:
            print(f"figure: {type(e).__name__}: {e}")
    summary(res)
    print(f"wrote tables to {GEN}, figures to {FIG}, summary to {RESULTS}/summary.md")


if __name__ == "__main__":
    main()
