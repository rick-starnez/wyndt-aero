# validate_sphere.py
"""
Validación del solver de paneles contra la solución analítica de una esfera.

La solución analítica para una esfera en flujo potencial es:
    Cp(θ) = 1 - (9/4) * sin²(θ)
donde θ es el ángulo desde el punto de estancamiento (0° = frente).

Este script:
1. Carga el STL de la esfera
2. Corre el solver con V∞ desde +X
3. Extrae el Cp de cada panel
4. Calcula el ángulo θ de cada panel respecto al punto de estancamiento
5. Agrupa por bins angulares
6. Compara contra la solución analítica
7. Genera la gráfica de validación
8. Guarda los datos en CSV
"""

import sys
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.panel_solver import PanelSolver3D


# ============================================================
# CONFIGURACIÓN
# ============================================================
SPHERE_STL = "sphere.stl"          # ← AJUSTA el nombre del archivo
V_INF = 1.0                        # m/s (usamos 1 para que Cp sea adimensional)
THETA = 0.0                        # ángulo azimutal del viento
PHI = 90.0                         # ángulo polar (viento desde +X)
N_ANGULAR_BINS = 37                # bins cada 5° (0°, 5°, 10°, ..., 180°)


def load_and_solve_sphere():
    """Carga la esfera y corre el solver."""
    print("=" * 60)
    print("🔵 SPHERE VALIDATION")
    print("=" * 60)
    
    if not os.path.exists(SPHERE_STL):
        print(f"❌ No se encontró: {SPHERE_STL}")
        print(f"   Copia el STL de la esfera a: {os.path.abspath(SPHERE_STL)}")
        return None
    
    # Cargar el STL
    print(f"\n📂 Cargando: {SPHERE_STL}")
    sphere = pv.read(SPHERE_STL)
    print(f"   Paneles: {sphere.n_cells}")
    
    # Calcular centro y radio
    bounds = sphere.bounds
    center = np.array([
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2
    ])
    radius = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4]
    ) / 2
    
    print(f"   Centro: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"   Radio:  {radius:.4f} m")
    
    # Correr el solver
    print(f"\n🔬 Corriendo el solver...")
    solver = PanelSolver3D(sphere, escala=1.0, rho=1.225, verbose=False)
    results = solver.solve(
        V_inf_magnitude=V_INF,
        theta=THETA,
        phi=PHI,
        use_gmres=False
    )
    
    # Extraer datos
    Cp = results['Cp']
    centers = solver.geometry.get_panel_centers()
    
    print(f"\n📊 Resultados del solver:")
    print(f"   Cp max:  {np.max(Cp):.4f}")
    print(f"   Cp min:  {np.min(Cp):.4f}")
    print(f"   Cp mean: {np.mean(Cp):.4f}")
    print(f"   Cp std:  {np.std(Cp):.4f}")
    
    return Cp, centers, center, radius


def compute_theta_from_stagnation(centers, center):
    """
    Calcula el ángulo θ de cada panel respecto al punto de estancamiento.
    
    El viento viene desde +X (θ=0°, φ=90°), así que el punto de estancamiento
    está en el punto más a la derecha de la esfera: (center_x + R, center_y, center_z).
    
    Para cada panel, θ es el ángulo entre:
      - el vector desde el centro al panel
      - el vector desde el centro al punto de estancamiento (dirección -X desde el panel)
    """
    # Vector unitario del viento (dirección desde donde viene el viento)
    # V_inf = V * [sin(φ)cos(θ), sin(φ)sin(θ), cos(φ)]
    # Con θ=0, φ=90: V_inf = V * [1, 0, 0], viene desde +X
    wind_dir = np.array([1.0, 0.0, 0.0])
    
    # Vectores desde el centro a cada panel
    r_vectors = centers - center
    
    # Normalizar
    r_norms = np.linalg.norm(r_vectors, axis=1)
    r_norms[r_norms < 1e-10] = 1.0
    r_hat = r_vectors / r_norms[:, np.newaxis]
    
    # Ángulo entre r_hat y wind_dir
    cos_theta = np.dot(r_hat, wind_dir)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # θ va de 0° (punto de estancamiento, frente) a 180° (punto trasero)
    theta_deg = np.degrees(np.arccos(cos_theta))
    
    return theta_deg


def bin_by_angle(theta_deg, Cp, n_bins=N_ANGULAR_BINS):
    """Agrupa el Cp por bins angulares."""
    bins = np.linspace(0, 180, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    Cp_binned = np.full(n_bins, np.nan)
    Cp_std_binned = np.full(n_bins, np.nan)
    n_panels_binned = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        mask = (theta_deg >= bins[i]) & (theta_deg < bins[i+1])
        n_panels_binned[i] = np.sum(mask)
        if n_panels_binned[i] > 0:
            Cp_binned[i] = np.mean(Cp[mask])
            Cp_std_binned[i] = np.std(Cp[mask])
    
    return bin_centers, Cp_binned, Cp_std_binned, n_panels_binned


def analytical_cp(theta_deg):
    """Solución analítica de la esfera: Cp(θ) = 1 - (9/4) sin²(θ)."""
    theta_rad = np.radians(theta_deg)
    return 1.0 - (9.0/4.0) * np.sin(theta_rad)**2


def plot_validation(theta_bins, Cp_binned, Cp_std, Cp_analytical, save_path=None):
    """Genera la gráfica de validación."""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Curva analítica
    theta_smooth = np.linspace(0, 180, 200)
    ax.plot(theta_smooth, analytical_cp(theta_smooth), 'k-', linewidth=2.5,
            label='Analytical Solution', zorder=1)
    
    # Datos del solver con barras de error
    ax.errorbar(theta_bins, Cp_binned, yerr=Cp_std,
                fmt='ro', markersize=6, capsize=3, capthick=1.5,
                label='Panel Method (Present)', zorder=2)
    
    ax.set_xlabel('Polar Angle θ (degrees)', fontsize=13)
    ax.set_ylabel('Pressure Coefficient $C_p$', fontsize=13)
    ax.set_title('Sphere Validation: $C_p$ vs Polar Angle', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    ax.set_xlim(0, 180)
    ax.set_ylim(-1.6, 1.4)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    # Calcular estadísticas
    mask_valid = ~np.isnan(Cp_binned)
    error = np.mean(np.abs(Cp_binned[mask_valid] - analytical_cp(theta_bins[mask_valid])))
    rmse = np.sqrt(np.mean((Cp_binned[mask_valid] - analytical_cp(theta_bins[mask_valid]))**2))
    max_error = np.max(np.abs(Cp_binned[mask_valid] - analytical_cp(theta_bins[mask_valid])))
    
    # Añadir cuadro de texto con estadísticas
    stats_text = (f'Mean |Error|: {error:.4f}\n'
                  f'RMSE: {rmse:.4f}\n'
                  f'Max |Error|: {max_error:.4f}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Gráfica guardada: {save_path}")
    
    plt.show()
    return error, rmse, max_error


def main():
    result = load_and_solve_sphere()
    if result is None:
        return
    
    Cp, centers, center, radius = result
    
    # Calcular θ de cada panel respecto al punto de estancamiento
    theta_deg = compute_theta_from_stagnation(centers, center)
    print(f"\n📐 Rango de θ: {np.min(theta_deg):.1f}° a {np.max(theta_deg):.1f}°")
    print(f"   Media de θ: {np.mean(theta_deg):.1f}°")
    
    # Agrupar por bins angulares
    theta_bins, Cp_binned, Cp_std, n_panels = bin_by_angle(theta_deg, Cp)
    Cp_analytical = analytical_cp(theta_bins)
    
    # Imprimir tabla de resultados
    print("\n" + "=" * 70)
    print(f"{'θ (°)':<8} {'Cp Num':<12} {'Cp Anal':<12} {'Error':<12} {'# Paneles':<10}")
    print("-" * 70)
    for i in range(len(theta_bins)):
        if not np.isnan(Cp_binned[i]):
            err = Cp_binned[i] - Cp_analytical[i]
            print(f"{theta_bins[i]:<8.1f} {Cp_binned[i]:<12.4f} "
                  f"{Cp_analytical[i]:<12.4f} {err:<12.4f} {n_panels[i]:<10d}")
    print("=" * 70)
    
    # Gráfica
    error, rmse, max_error = plot_validation(
        theta_bins, Cp_binned, Cp_std, Cp_analytical,
        save_path='figure_5_sphere_validation.png'
    )
    
    # Guardar CSV
    np.savetxt(
        'sphere_cp_validation_new.csv',
        np.column_stack([theta_bins, Cp_analytical, Cp_binned, Cp_std, n_panels]),
        delimiter=',',
        header='Theta_deg,Cp_Analytical,Cp_Numerical,Cp_Std,N_Panels',
        comments='',
        fmt='%.6f'
    )
    print(f"✅ CSV guardado: sphere_cp_validation_new.csv")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE VALIDACIÓN")
    print("=" * 60)
    print(f"   Error medio absoluto:    {error:.4f}")
    print(f"   RMSE:                    {rmse:.4f}")
    print(f"   Error máximo absoluto:   {max_error:.4f}")
    print(f"\n   Interpretación:")
    if error < 0.05:
        print("   ✅ EXCELENTE: Error medio < 5%")
    elif error < 0.10:
        print("   ✅ BUENO: Error medio < 10%")
    elif error < 0.20:
        print("   ⚠️ ACEPTABLE: Error medio < 20%")
    else:
        print("   ❌ INSUFICIENTE: Error medio > 20%")
    print("=" * 60)


if __name__ == "__main__":
    main()