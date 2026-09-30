# test_gui_integration.py
"""
Prueba rápida para verificar que la GUI funciona con el nuevo solver
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.panel_solver import PanelSolver3D
import pyvista as pv

print("✅ Importación correcta")

# Cargar un rotor sintético
rotor = pv.read("rotor_sintetico.stl")
print(f"   Rotor cargado: {rotor.n_cells} paneles")

# Crear solver
solver = PanelSolver3D(rotor, scale=1.0, rho=1.225, verbose=True)

# Resolver
resultados = solver.solve(
    V_inf_magnitude=16.0,
    theta=0,
    phi=90,
    use_gmres=(rotor.n_cells > 500)
)

print(f"\n📊 RESULTADOS:")
print(f"   Cl: {resultados['forces']['Cl']:.6f}")
print(f"   Cd: {resultados['forces']['Cd']:.6f}")
print(f"   Torque: {resultados['forces']['torque_z']:.6f} N·m")

print("\n✅ Solver funcionando correctamente")