# benchmark_direct_vs_gmres.py
"""
Compara solver directo vs GMRES para diferentes tamaños de malla.
"""

import sys
import os
import time
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.panel_solver import PanelSolver3D


def benchmark_both(rotor_file, n_angles=5):
    print("=" * 70)
    print(f"⏱️  COMPARACIÓN: DIRECT vs GMRES")
    print(f"   Archivo: {rotor_file}")
    print("=" * 70)
    
    rotor = pv.read(rotor_file)
    print(f"\n📊 Paneles: {rotor.n_cells}")
    
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    
    # --- Solver directo ---
    print(f"\n🔷 SOLVER DIRECTO (LU)")
    solver_direct = PanelSolver3D(rotor, escala=1.0, rho=1.225, verbose=False)
    _ = solver_direct._get_influence_data()  # Pre-construir matriz
    
    t0 = time.time()
    for theta in angles:
        _ = solver_direct.solve(V_inf_magnitude=16.0, theta=theta, phi=90.0, 
                                 use_gmres=False)
    t_direct = time.time() - t0
    print(f"   Tiempo total ({n_angles} ángulos): {t_direct:.3f} s")
    print(f"   Tiempo por ángulo:                 {t_direct/n_angles:.3f} s")
    
    # --- Solver GMRES ---
    print(f"\n🔶 SOLVER GMRES (iterativo)")
    solver_gmres = PanelSolver3D(rotor, escala=1.0, rho=1.225, verbose=False)
    _ = solver_gmres._get_influence_data()  # Pre-construir matriz
    
    t0 = time.time()
    for theta in angles:
        _ = solver_gmres.solve(V_inf_magnitude=16.0, theta=theta, phi=90.0, 
                                use_gmres=True)
    t_gmres = time.time() - t0
    print(f"   Tiempo total ({n_angles} ángulos): {t_gmres:.3f} s")
    print(f"   Tiempo por ángulo:                 {t_gmres/n_angles:.3f} s")
    
    # --- Comparación ---
    print("\n" + "=" * 70)
    print("📊 COMPARACIÓN FINAL")
    print("=" * 70)
    print(f"   Directo:  {t_direct:.3f} s ({t_direct/n_angles:.3f} s/ángulo)")
    print(f"   GMRES:    {t_gmres:.3f} s ({t_gmres/n_angles:.3f} s/ángulo)")
    
    if t_gmres < t_direct:
        speedup = t_direct / t_gmres
        print(f"\n   ✅ GMRES es {speedup:.2f}× más rápido")
        print(f"   → Para un barrido de 18 ángulos: {t_gmres/n_angles*18:.1f} s vs {t_direct/n_angles*18:.1f} s")
    else:
        speedup = t_gmres / t_direct
        print(f"\n   ⚠️ El solver directo es {speedup:.2f}× más rápido")
        print(f"   → El umbral de 1000 paneles en el código puede necesitar ajuste")
    
    print("=" * 70)
    
    return t_direct, t_gmres


if __name__ == "__main__":
    benchmark_both("rotor_2_fixed.stl", n_angles=5)