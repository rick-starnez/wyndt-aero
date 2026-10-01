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

- BEM (Blade Element Momentum): Fast but relies on empirical corrections.
- CFD (Computational Fluid Dynamics): High fidelity but computationally expensive.

Panel methods offer a middle ground. WyndT addresses the gap by providing:

- A robust panel method solver optimized with Numba.
- An intuitive graphical user interface.
- Automated diagnostics to detect problematic configurations.

Key Features

## Key Features

- 3D panel method solver based on constant-strength source distributions.
- Numba-accelerated influence matrix assembly with parallel execution.
- Influence matrix caching for fast azimuthal sweeps (up to 23x speedup).
- Automatic STL import with unit detection (mm/m) and mesh repair.
- Interactive 3D visualization of pressure coefficient (Cp) distributions.
- Dynamic camera that orients according to the incoming flow direction.
- Rotor diagnostics module with six complementary visualizations.
- Azimuthal sweep to generate Cl, Cd, L/D, and torque curves.
- Built-in symmetry detection to identify degenerate configurations.
- Cross-platform: tested on macOS (Apple Silicon), compatible with Linux and Windows.

Installation
Requirements

-Python 3.10 or higher
- Operating System: macOS, Linux, or Windows
- Recommended hardware: multi-core CPU 

Dependencies

The following Python packages are required:

- numpy: Numerical arrays
- scipy: Linear algebra solvers
- numba: JIT compilation and parallelization
- pyvista: 3D visualization
- pyvistaqt: PyVista integration with Qt
- pyqt6: Graphical user interface
- matplotlib: 2D plotting

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

- Click "Load Wind Turbine STL" and select your rotor geometry.
- Adjust the Polar Angle (phi) if needed.
- Click "Update View" to recompute the flow and update the 3D visualization.
- Click "Plot Aerodynamic Curves" to run a full azimuthal sweep.
- Click "Rotor Diagnostics" to generate the diagnostic plots.

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

- Load Wind Turbine STL: Opens a file dialog to load the geometry.
- Plot Aerodynamic Curves: Runs the azimuthal sweep.
- Rotor Diagnostics: Generates 6 diagnostic plots.
- Polar Angle (phi): Adjusts the flow direction.
- Update View: Recomputes the flow for the current phi.

Right area (3D viewer):

- Displays the pressure coefficient (Cp) distribution.
- The camera orients according to the flow direction.
- A color bar indicates the Cp range.

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

- 3D View of Rotor - Scatter plot of panel centroids.
- Angular Distribution of Cells - Histogram of azimuthal angles.
- Distribution in Z (Thickness) - Histogram of thickness.
- Cp Distribution - 2D map of the pressure coefficient.
- Torque per Panel - 2D map of torque contribution.
- Torque per Angular Sector - Bar chart by sector.

These visualizations help detect:

- Excessive symmetry.
- Torque cancellation.
- Numerical artifacts.

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

## Performance

Benchmarks were conducted on an Apple Mac Studio with an M5 Max chip (18 cores, 36 GB RAM), using NumPy and SciPy linked against the Accelerate framework.

### Representative results (4-blade asymmetric rotor, 9,216 panels)

| Operation | Time |
|-----------|------|
| STL loading | 0.22 s |
| Solver initialization | 0.55 s |
| Influence matrix assembly (once) | 0.83 s |
| Single flow solve (with cached matrix) | 2.68 s |
| Full azimuthal sweep (5° resolution, 73 points) | 196 s (~3.3 min) |
| Full azimuthal sweep (2° resolution, 181 points) | ~8 min |

### Scaling with number of panels

| Panels | Matrix assembly | Single solve |
|--------|----------------|--------------|
| 2,000 | ~1 s | ~0.5 s |
| 4,000 | ~4 s | ~1.2 s |
| 9,216 | ~0.8 s | ~2.7 s |

Note: matrix assembly time does not scale monotonically because Numba's parallelization efficiency depends on the matrix size and available cores.

### Key optimizations

- **Numba parallelization**: Matrix assembly uses all available CPU cores (18 threads on M5 Max).
- **Influence matrix caching**: The matrix depends only on geometry, so it is reused across all azimuthal angles. This reduces the cost of an 18-point sweep from ~5 minutes (naive) to ~48 seconds — a speedup of about 6x.
- **Direct solver vs. GMRES**: For dense matrices, the direct LU decomposition (Accelerate BLAS-3) is **23.6x faster** than the iterative GMRES solver (BLAS-2). Based on this benchmark, the solver is configured to always use the direct method.

## Known Limitations

WyndT is based on the incompressible, irrotational, inviscid potential flow formulation. This means:

- No viscous effects: Drag predictions omit skin friction.
- No flow separation: Lift is over-predicted beyond the stall angle.
- Steady flow only: Transient phenomena are not captured.
- Watertight geometry required: Non-manifold edges can cause numerical issues.
- Source-only formulation: The solver uses constant-strength sources. This is adequate for 2D airfoils and rotor configurations, but exhibits systematic errors for fully 3D bodies such as a sphere.

### Symmetry-induced cancellation

A rotor with perfect rotational symmetry produces zero net lift and drag, and a constant torque independent of azimuthal angle. This is a mathematically correct result of the potential flow formulation, not a bug: the perfect rotational symmetry of the geometry causes a complete cancellation of the pressure forces when summed over all blades.

This behavior should be interpreted carefully:

- **It does not occur in real rotors.** Manufacturing tolerances, blade wear, and natural wind turbulence always break the symmetry to some degree. A physical rotor with nominally symmetric blades will still rotate and produce variable torque.
- **It is a valid diagnostic case.** The solver's ability to reproduce this analytically expected result is itself a form of validation.
- **It highlights the importance of asymmetry.** Any meaningful rotor design must break rotational symmetry — through non-uniform blade spacing, different blade geometries, or asymmetric airfoil sections — to generate useful aerodynamic forces.

To analyze asymmetric rotors effectively, users should ensure that:

1. The blade angular positions are not multiples of 360°/n, where n is the number of blades.
2. The blades are not geometrically identical (e.g., different chord, twist, or span distributions).
3. The simulation captures the full 360° azimuthal sweep, since the symmetric case would give a trivial result.

The diagnostics module automatically detects and warns about symmetric configurations.

### Numerical artifacts

Local values of Cp outside the physical range (e.g., Cp < -10) may appear even for well-conditioned meshes. These are usually numerical artifacts concentrated in regions of high curvature. The diagnostics module uses percentile-based color scaling to suppress their visual impact.

### Drag sign convention

The drag force is reported as a magnitude in the Cd output, since its sign depends on the direction of the freestream relative to the asymmetric rotor geometry. The signed value is retained in the solver output as `drag_signed` for diagnostic purposes. For highly asymmetric rotors, the signed drag can reverse sign at certain azimuthal angles, indicating that the rotor is pushing the flow rather than being pushed by it.

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
