# NSC peak near the percolation threshold — code and data

Code and data for the paper

> I. A. Nakonechnyi, A. A. Snarskii, *Peak of network structural complexity near the percolation threshold:
> when does it indicate the transition* (preprint in preparation; Ukrainian draft).

The paper studies the network structural complexity (NSC) of A. A. Snarskii
[Phys. Rev. E **110**, 054309 (2024)] near the percolation threshold: finite-size scaling of the NSC peak on
square, triangular and hypercubic lattices d = 2…6 (up to 1.7·10⁷ sites, sampled estimator), exact mean-field
results via Galton–Watson trees (Erdős–Rényi, random regular graphs, Bethe lattices, clique trees), and the
singularity of NSC at the threshold.

## Layout

| Path | Contents |
| --- | --- |
| `code/lib/` | library: exact NSC on numba (`nsccore`), sampled estimator (`sampled`), percolation generators (`perc`, `lattices3d`), statistics (`stats`), exact mean-field NSC on Galton–Watson trees (`gw`, `gwexact`, `gwtree`, `treeenum`, `ledger`) and on clique trees (`cactus`, `cactusexact`), run metadata (`runmeta`); tests in `code/lib/tests` |
| `code/P1/` | scripts that produce every result of the paper (`p1_*_run.py` — computations, `p1_*_analyse.py` — analysis) |
| `results/P1/` | result JSON files; each has a `meta` block (script, sha256 of the script, seeds, parameters, package versions, git commit of the authors' working repository) |
| `notes/orchestra/` | independent re-checks of selected results (scripts and their JSON output) |
| `data/P1/constants.csv` | thresholds and critical exponents with a literature source for every number |
| `paper/P1/make_figures.py` | builds all figures of the paper and `numbers.json` (every derived number with its source) from `results/P1` |

## Reproduce

Python 3.12; `pip install -r requirements.txt`.

```bash
python paper/P1/make_figures.py          # figures -> paper/P1/figs/, derived numbers -> paper/P1/numbers.json (seconds)
python -m pytest code/lib/tests -q       # tests
python code/P1/p1_3c_run.py              # example of a full computation: exact NSC peak on clique trees (~45 min)
```

The large finite-size-scaling runs (`code/P1/p1_3_run.py`, ~110 CPU hours on a 10-core Apple M4) write
checkpoints to `data/P1/` (not included); their summaries are in `results/P1/p1_3/`.

## License

MIT (see `LICENSE`).
