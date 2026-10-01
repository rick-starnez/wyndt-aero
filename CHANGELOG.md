# Changelog

All notable changes to WyndT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Symmetry detection in the diagnostics module: the tool now warns when the rotor has perfect rotational symmetry, which produces zero net lift and drag.
- New documentation section on symmetry-induced cancellation in the README.
- New drag sign convention: `drag_magnitude` (always positive) is now the default output, while `drag_signed` is retained for diagnostic purposes.

### Fixed

- Fixed a bug in the drag calculation where the signed drag component was used inconsistently in the lift vector computation, causing incorrect lift values when the drag reversed sign for asymmetric rotors.
- The Cd coefficient is now always reported as a positive magnitude, consistent with standard aerodynamic convention.

### Changed

- Solver is now configured to use the direct LU solver by default for all problem sizes, after benchmarks showed it is 23.6x faster than GMRES for the dense influence matrices encountered in this work.
- README has been reorganized with clearer sections on limitations and performance.

## [0.1.0-beta] - 2026-09-30

### Added

- Initial public release.
- 3D panel method solver with Numba-accelerated influence matrix assembly.
- Influence matrix caching for fast azimuthal sweeps (up to 23x speedup).
- Automatic STL import with unit detection (mm/m) and mesh repair.
- Interactive 3D visualization of pressure coefficient (Cp) distributions.
- Dynamic camera that orients according to the incoming flow direction.
- Rotor diagnostics module with six complementary visualizations.
- Azimuthal sweep to generate Cl, Cd, L/D, and torque curves.

[Unreleased]: https://github.com/rick-starnez/wyndt-aero/compare/v0.1.0-beta...HEAD
[0.1.0-beta]: https://github.com/rick-starnez/wyndt-aero/releases/tag/v0.1.0-beta
