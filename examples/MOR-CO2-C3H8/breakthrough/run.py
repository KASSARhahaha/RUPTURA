from __future__ import annotations

import pathlib
import sys

import matplotlib


ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend configured above)

import ruptura  # noqa: E402  (import after sys.path adjustment)


def main() -> None:
    base_dir = pathlib.Path(__file__).resolve().parent
    ruptura_objs = ruptura.from_input(base_dir / "simulation.input")
    ruptura_objs["Breakthrough"].compute()
    fig, ax = plt.subplots()
    ruptura_objs["Breakthrough"].plot(ax, "breakthrough")

    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    fig.savefig(results_dir / "breakthrough.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
