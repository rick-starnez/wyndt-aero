WyndT

A Python-based panel method tool for rapid aerodynamic analysis of wind turbine rotors.

https://img.shields.io/badge/Python-3.10%2B-blue.svg
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/Status-Beta-orange.svg
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23085462.svg)](https://doi.org/10.5281/zenodo.23085462)

WyndT is an open-source Python tool for preliminary aerodynamic analysis of wind turbine rotors using a 3D panel method.
Table of Contents

    Motivation

    Key Features

    Installation

    Quick Start

    Usage

    Diagnostics Module

    Architecture

    Performance

    Known Limitations

    Citation

    License

    Contact

Motivation

Wind turbine blade design requires accurate aerodynamic analysis to maximize energy capture while minimizing structural loads. Traditional approaches face a trade-off:

    BEM (Blade Element Momentum): Fast but relies on empirical corrections.

    CFD (Computational Fluid Dynamics): High fidelity but computationally expensive.

Panel methods offer a middle ground. WyndT addresses the gap by providing:

    A robust panel method solver optimized with Numba.

    An intuitive graphical user interface.

    Automated diagnostics to detect problematic configurations.

Key Features

    3D panel method solver based on constant-strength source distributions.

    Numba-accelerated influence matrix assembly with parallel execution.

    Influence matrix caching for fast azimuthal sweeps (up to 23x speedup).

    Automatic STL import with unit detection and mesh repair.

    Interactive 3D visualization of pressure coefficient (Cp) distributions.

    Dynamic camera that orients according to the incoming flow direction.

    Rotor diagnostics module with six complementary visualizations.

    Azimuthal sweep to generate Cl, Cd, L/D, and torque curves.

Installation
Requirements

    Python 3.10 or higher

    Operating System: macOS, Linux, or Windows

    Recommended hardware: multi-core CPU

Dependencies

The following Python packages are required:

    numpy: Numerical arrays

    scipy: Linear algebra solvers

    numba: JIT compilation and parallelization

    pyvista: 3D visualization

    pyvistaqt: PyVista integration with Qt

    pyqt6: Graphical user interface

    matplotlib: 2D plotting

Installation Steps

    Clone the repository:

bash

git clone https://github.com/rick-starnez/wyndt-aero.git
cd wyndt-aero

    Create a virtual environment (recommended):

bash

python -m venv venv
source venv/bin/activate

On Windows, use:
bash

venv\Scripts\activate

    Install dependencies:

bash

pip install -r requirements.txt

    Verify the installation:

bash

python -c "from src.core.panel_solver import PanelSolver3D; print('WyndT ready')"

Quick Start
Graphical User Interface

Launch the GUI:
bash

python STL07C_paneles.py

Then:

    Click "Load Wind Turbine STL" and select your rotor geometry.

    Adjust the Polar Angle (phi) if needed.

    Click "Update View" to recompute the flow and update the 3D visualization.

    Click "Plot Aerodynamic Curves" to run a full azimuthal sweep.

    Click "Rotor Diagnostics" to generate the diagnostic plots.

Python API

For scripting or batch processing:
python

import pyvista as pv
from src.core.panel_solver import PanelSolver3D

rotor = pv.read("rotor_2_fixed.stl")

solver = PanelSolver3D(rotor, escala=1.0, rho=1.225)

results = solver.solve(
    V_inf_magnitude=16.0,
    theta=0.0,
    phi=90.0,
    use_gmres=False
)

print(f"Cl       = {results['forces']['Cl']:.4f}")
print(f"Cd       = {results['forces']['Cd']:.4f}")
print(f"Torque   = {results['forces']['torque_z']:.4f} N-m")
print(f"Solution time: {results['solution_time']:.2f} s")

Usage
Graphical User Interface

The GUI is organized into two main areas:

Left panel (controls):

    Load Wind Turbine STL: Opens a file dialog to load the geometry.

    Plot Aerodynamic Curves: Runs the azimuthal sweep.

    Rotor Diagnostics: Generates 6 diagnostic plots.

    Polar Angle (phi): Adjusts the flow direction.

    Update View: Recomputes the flow for the current phi.

Right area (3D viewer):

    Displays the pressure coefficient (Cp) distribution.

    The camera orients according to the flow direction.

    A color bar indicates the Cp range.

Python API

Basic solver usage:
python

from src.core.panel_solver import PanelSolver3D
import pyvista as pv

rotor = pv.read("rotor.stl")
solver = PanelSolver3D(rotor, escala=1.0, rho=1.225)
results = solver.solve(V_inf_magnitude=16.0, theta=0.0, phi=90.0)

Azimuthal sweep:
python

import numpy as np

angles = np.linspace(0, 360, 18)
torques = []

solver = PanelSolver3D(rotor, escala=1.0, rho=1.225)
for theta in angles:
    res = solver.solve(V_inf_magnitude=16.0, theta=theta, phi=90.0)
    torques.append(res['forces']['torque_z'])

print(f"Torque range: {min(torques):.4f} to {max(torques):.4f} N-m")

Diagnostics Module

The diagnostics module generates six visualizations:

    3D View of Rotor - Scatter plot of panel centroids.

    Angular Distribution of Cells - Histogram of azimuthal angles.

    Distribution in Z (Thickness) - Histogram of thickness.

    Cp Distribution - 2D map of the pressure coefficient.

    Torque per Panel - 2D map of torque contribution.

    Torque per Angular Sector - Bar chart by sector.

These visualizations help detect:

    Excessive symmetry.

    Torque cancellation.

    Numerical artifacts.

Architecture

The project structure is:
text

wyndt-aero/
    STL07C_paneles.py
    src/
        core/
            panel_solver.py
            panel_geometry.py
            panel_utils_new.py
    examples/
        rotor_sintetico.stl
        rotor_2_fixed.stl
    tests/
    requirements.txt
    LICENSE
    README.md

Performance

Benchmarks on an Apple Mac Studio with M5 Max (18 cores, 36 GB RAM):

    STL loading: 0.22 s

    Solver initialization: 0.55 s

    Influence matrix assembly: 0.83 s

    Single flow solve: 2.68 s

    Full azimuthal sweep (18 points): 48.3 s

Key optimizations:

    Numba parallelization across all CPU cores.

    Influence matrix caching reused across angles.

    Direct LU solver (23.6x faster than GMRES for dense matrices).

Known Limitations

WyndT assumes incompressible, irrotational, inviscid flow:

    No viscous effects.

    No flow separation.

    Steady flow only.

    Watertight geometry required.

    Source-only formulation.

Symmetry-induced torque cancellation

A rotor with perfect rotational symmetry produces constant torque. This is mathematically correct but uninformative for engineering.
Numerical artifacts

Local Cp values outside the physical range may appear. The diagnostics module uses percentile-based color scaling.

## Citation

If you use WyndT in your research, please cite both the software and the archived version.

### Software (preferred citation)

@software{wyndt2026,
  title = {WyndT: A Python-based panel method tool for aerodynamic analysis of wind turbine rotors},
  author = {Martínez González, Ricardo Francisco},
  year = {2026},
  version = {0.1.0-beta},
  doi = {10.5281/zenodo.23085462},
  url = {https://doi.org/10.5281/zenodo.23085462},
  institution = {Tecnológico Nacional de México - IT de Veracruz}
}

### Repository

https://github.com/rick-starnez/wyndt-aero

A full paper describing the tool's architecture and validation is in preparation.
License

This project is licensed under the MIT License. See the LICENSE file for details.
Contact

Author: Ricardo Francisco Martinez Gonzalez
Institution: Tecnologico Nacional de Mexico - Instituto Tecnologico de Veracruz
Email: rick.starnez@gmail.com
GitHub: @rick-starnez

For bug reports, please open an issue at:

https://github.com/rick-starnez/wyndt-aero/issues
Acknowledgments

    Panel method formulation based on Katz & Plotkin, Low-Speed Aerodynamics.

    GUI built on PyQt6 and PyVista.

    Influence matrix accelerated by Numba.

    BLAS/LAPACK through Apple's Accelerate framework.

Status: Beta
Last updated: September 2026.
