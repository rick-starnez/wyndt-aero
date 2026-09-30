# benchmark_solver.py
"""
Benchmark del solver: mide el tiempo de cada fase.
"""

import sys
import os
import time
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.panel_solver import PanelSolver3D


def benchmark(rotor_file, n_angles=18):
    print("=" * 60)
    print(f"⏱️  BENCHMARK: {rotor_file}")
    print("=" * 60)
    
    # Cargar
    t0 = time.time()
    rotor = pv.read(rotor_file)
    t_load = time.time() - t0
    print(f"\n📂 Carga de STL:            {t_load:.3f} s")
    print(f"   Paneles: {rotor.n_cells}")
    
    # Inicializar solver (incluye PanelGeometry)
    t0 = time.time()
    solver = PanelSolver3D(rotor, escala=1.0, rho=1.225, verbose=False)
    t_init = time.time() - t0
    print(f"\n🔧 Inicialización:          {t_init:.3f} s")
    
    # Construcción de la matriz (primera vez)
    t0 = time.time()
    _ = solver._get_influence_data()
    t_matrix = time.time() - t0
    print(f"🔨 Matriz de influencia:    {t_matrix:.3f} s")
    
    # Primer solve (usa el caché)
    t0 = time.time()
    _ = solver.solve(V_inf_magnitude=16.0, theta=0.0, phi=90.0, use_gmres=False)
    t_first = time.time() - t0
    print(f"⚡ Primer solve:            {t_first:.3f} s")
    
    # Barrido (con caché)
    angles = np.linspace(0, 360, n_angles)
    t0 = time.time()
    for theta in angles:
        _ = solver.solve(V_inf_magnitude=16.0, theta=theta, phi=90.0, use_gmres=False)
    t_sweep = time.time() - t0
    t_per_angle = t_sweep / n_angles
    print(f"\n🔄 Barrido ({n_angles} ángulos):")
    print(f"   Tiempo total:            {t_sweep:.3f} s")
    print(f"   Tiempo por ángulo:       {t_per_angle:.3f} s")
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    print(f"   Carga + Init:            {t_load + t_init:.3f} s")
    print(f"   Matriz (una vez):        {t_matrix:.3f} s")
    print(f"   Sweep ({n_angles} pts):        {t_sweep:.3f} s")
    print(f"   TOTAL:                   {t_load + t_init + t_matrix + t_sweep:.3f} s")
    print("=" * 60)
    
    return {
        't_load': t_load,
        't_init': t_init,
        't_matrix': t_matrix,
        't_first': t_first,
        't_sweep': t_sweep,
        't_per_angle': t_per_angle,
        'n_panels': rotor.n_cells,
    }


if __name__ == "__main__":
    rotor_file = "rotor_2_fixed.stl"  # ← ajusta si es necesario
    
    if not os.path.exists(rotor_file):
        print(f"❌ No se encontró: {rotor_file}")
        print("   Ajusta el nombre del archivo en el script.")
        sys.exit(1)
    
    results = benchmark(rotor_file, n_angles=18)
    
    # Opcional: guardar resultados en CSV para comparaciones futuras
    import csv
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, v])
    print(f"\n💾 Resultados guardados en: benchmark_results.csv")