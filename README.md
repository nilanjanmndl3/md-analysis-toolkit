# MD Analysis Toolkit

A Python toolkit for analysing molecular dynamics (MD) trajectories and extracting structural and transport properties.

## Features

- Mean square displacement (MSD)
- Diffusion coefficient using the Einstein relation
- Radial distribution function (RDF)
- One-dimensional number-density profile
- Reproducible example using synthetic particle trajectories
- Unit tests with `pytest`

## Project status

This is the first development version. The initial implementation uses NumPy arrays and synthetic particle trajectories. Trajectory-file readers (for example, GROMACS `.xtc`/`.trr` and LAMMPS dump files) can be added in later versions.

## Installation

```bash
git clone https://github.com/nilanjanmndl3/md-analysis-toolkit.git
cd md-analysis-toolkit
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install the package and development dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

Run the example:

```bash
python examples/run_analysis.py
```

The script generates a random-walk trajectory, calculates the MSD and RDF, estimates the diffusion coefficient, and saves figures in `results/figures/`.

## Input format

The core functions accept positions as a NumPy array with shape:

`(n_frames, n_particles, 3)`

Positions should be **unwrapped** for MSD calculations. The RDF function in this initial version assumes a cubic periodic box and applies the minimum-image convention.

## Scientific background

For three-dimensional diffusion, the Einstein relation is:

\[
D = \lim_{t\to\infty}\frac{1}{6}\frac{d}{dt}\langle|\mathbf r_i(t)-\mathbf r_i(0)|^2\rangle
\]

The RDF, \(g(r)\), describes the relative probability of finding another particle at a distance \(r\), compared with an ideal uniform distribution at the same number density.

## Tests

```bash
pytest
```

## Planned development

- [ ] Add GROMACS trajectory reader
- [ ] Add LAMMPS dump reader
- [ ] Support triclinic and orthorhombic periodic boxes
- [ ] Add time-origin averaging for MSD
- [ ] Add block-averaged uncertainty estimates
- [ ] Add atom-selection and species-selection utilities
- [ ] Add examples based on public benchmark data

## Repository structure

```text
md-analysis-toolkit/
├── src/mdtoolkit/
│   ├── __init__.py
│   ├── transport.py
│   ├── structure.py
│   └── profiles.py
├── examples/
│   └── run_analysis.py
├── tests/
│   ├── test_transport.py
│   └── test_structure.py
├── results/figures/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Author

**Nilanjan Mondal**  
Computational scientist | Molecular simulation | Multiscale modelling | CFD

GitHub: https://github.com/nilanjanmndl3

## License

MIT
