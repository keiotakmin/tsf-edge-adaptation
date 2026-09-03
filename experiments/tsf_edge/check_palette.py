"""Colour-blind safety check for the figure palettes (paper-figs / dataviz rule: COMPUTE the
separation, never eyeball it). The bundled dataviz validator is a node script and this host has
no node, so the same six-check logic is reimplemented here in numpy.

Pipeline: sRGB -> linear -> (Vienot-Brettel-Mollon LMS dichromat simulation) -> OKLab, then
pairwise Euclidean distance in OKLab x100 ("dE").

Thresholds (from the dataviz skill):
    normal vision  dE >= 15   hard requirement -- below it full-colour readers cannot separate
    CVD            dE >= 8    target;  6-8 legal ONLY with a secondary encoding (shape/label)

Run: .venv/bin/python experiments/tsf_edge/check_palette.py
"""
from __future__ import annotations
import itertools
import sys

import numpy as np

# linear RGB -> LMS (Hunt-Pointer-Estevez as used by Vienot et al. 1999)
RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                    [3.45565, 27.1554, 3.86714],
                    [0.0299566, 0.184309, 1.46709]])
LMS2RGB = np.linalg.inv(RGB2LMS)
SIM = {  # dichromat projections in LMS
    "deuteranopia": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
    "protanopia":   np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
    "tritanopia":   np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]),
}


def hex2lin(h):
    c = np.array([int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def simulate(lin, kind):
    return LMS2RGB @ (SIM[kind] @ (RGB2LMS @ lin))


def oklab(lin):
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return m2 @ np.cbrt(np.clip(m1 @ lin, 0, None))


def report(palette, names=None, secondary_encoding=True):
    names = names or list(palette)
    lin = {n: hex2lin(h) for n, h in zip(names, palette)}
    worst, fails, warns = {}, [], []
    for view in ["normal"] + list(SIM):
        lab = {n: oklab(v if view == "normal" else simulate(v, view)) for n, v in lin.items()}
        d = {(a, b): 100 * float(np.linalg.norm(lab[a] - lab[b]))
             for a, b in itertools.combinations(names, 2)}
        pair, val = min(d.items(), key=lambda kv: kv[1])
        worst[view] = (pair, val)
        floor = 15.0 if view == "normal" else 8.0
        soft = 15.0 if view == "normal" else 6.0
        tag = "PASS"
        if val < soft:
            tag = "FAIL"; fails.append((view, pair, val))
        elif val < floor:
            tag = "WARN" if secondary_encoding else "FAIL"
            (warns if secondary_encoding else fails).append((view, pair, val))
        print(f"  {view:13s} worst pair {pair[0]:>14s} / {pair[1]:<14s} dE = {val:5.1f}   [{tag}]")
    return fails, warns


if __name__ == "__main__":
    # Validate the palettes the figures ACTUALLY draw, by importing their definitions rather
    # than restating them here. The previous version hard-coded a hand-copied list, which is
    # how it came to keep reporting the retired sgd-blue / adafactor-purple pair (dE 2.0 under
    # protanopia) long after that combination stopped being what any figure used -- a checker
    # that can disagree with the thing it checks is worse than none.
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import matplotlib
    matplotlib.use("Agg")
    from stage0_figs import SERIES, FAM

    all_fails = []

    print("lr_response_paper series palette (colour = optimizer identity; marker shape is a "
          "redundant second encoding):")
    names = [k for k, *_ in SERIES]
    fails, warns = report([c for _, _, c, _ in SERIES], names)
    all_fails += [("lr_response", *f) for f in fails]

    print("\nFamily palette (colour = FAMILY in every figure; every point also carries a "
          "distinct marker and a direct text label, which is what licenses a 6-8 dE pair):")
    fails, warns = report([c for c, _ in FAM.values()], list(FAM))
    all_fails += [("tradeoff", *f) for f in fails]
    if warns:
        print("\n  WARN pairs rely on the secondary encoding (per-point text labels + distinct "
              "markers), which this figure always draws.")

    if all_fails:
        print("\n  FAIL:", "; ".join(f"[{fig}] {v} {a}/{b} dE={d:.1f}"
                                     for fig, v, (a, b), d in all_fails))
        sys.exit(1)
    print("\n  all palettes OK")
