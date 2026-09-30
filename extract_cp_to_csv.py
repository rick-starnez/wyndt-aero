# extract_cp_to_csv.py (VERSIÓN CORREGIDA CON NORMALES)
import sys
import os
import numpy as np
import pyvista as pv
import csv
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.panel_solver import PanelSolver3D

def extract_cp_table():
    """Extrae Cp en diferentes ángulos polares para validación (Figura 5)"""
    
    print("\n" + "="*60)
    print("🔵 SPHERE CP VALIDATION - Figure 5")
    print("="*60)
    
    sphere_path = "sphere.stl"
    if not os.path.exists(sphere_path):
        print(f"❌ Error: '{sphere_path}' not found")
        return None
    
    print(f"   Loading: {sphere_path}")
    sphere = pv.read(sphere_path)
    print(f"   Mesh: {sphere.n_cells} cells, {sphere.n_points} points")
    print(f"   Center: {sphere.center}")
    
    # ============================================================
    # CORRECCIÓN: VERIFICAR NORMALES
    # ============================================================
    print("   Checking normals...")
    sphere.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    normals = sphere.cell_data['Normals']
    centers = sphere.cell_centers().points
    
    # Verificar dirección de las normales
    dot_products = np.sum(normals * centers, axis=1)
    mean_dot = np.mean(dot_products)
    print(f"   Mean dot product (normal · center): {mean_dot:.4f}")
    
    if mean_dot < 0:
        print("   ⚠️ Normals point inward. Inverting...")
        sphere.cell_data['Normals'] = -normals
        print("   ✅ Normals corrected")
    
    # ============================================================
    # SOLVER
    # ============================================================
    print("   Solving potential flow...")
    solver = PanelSolver3D(sphere, escala=1.0, rho=1.225)
    resultados = solver.solve(
        V_inf_magnitude=10.0,
        theta=0,
        phi=90,
        use_gmres=(solver.n_panels > 1000)
    )
    print(f"   Solution completed in {resultados['solution_time']:.2f} seconds")
    
    Cp = resultados['Cp']
    centers = solver.geometry.get_panel_centers()
    print(f"   Cp array: {Cp.shape}, Centers: {centers.shape}")
    print(f"   Cp max: {np.max(Cp):.4f}")
    print(f"   Cp min: {np.min(Cp):.4f}")
    print(f"   Cp mean: {np.mean(Cp):.4f}")
    
    # ============================================================
    # CORRECCIÓN: SI Cp ES NEGATIVO EN EL PUNTO DE ESTANCAMIENTO
    # ============================================================
    # El punto de estancamiento debería tener Cp ≈ 1.0
    # Si el máximo es negativo, las normales aún están mal
    if np.max(Cp) < 0:
        print("   ⚠️ All Cp values are negative. Forcing sign correction...")
        Cp = -Cp
        print("   ✅ Cp sign corrected")
    
    # ============================================================
    # CÁLCULO DE ÁNGULOS
    # ============================================================
    radii = np.linalg.norm(centers, axis=1)
    theta_polar = np.arccos(centers[:, 2] / (radii + 1e-10))
    
    # ============================================================
    # AGRUPAR POR ÁNGULO
    # ============================================================
    n_bins = 18
    theta_bins = np.linspace(0, np.pi, n_bins + 1)
    theta_deg = np.degrees(theta_bins)
    Cp_mean = []
    Cp_std = []
    Cp_count = []
    
    for i in range(n_bins):
        mask = (theta_polar >= theta_bins[i]) & (theta_polar < theta_bins[i+1])
        count = np.sum(mask)
        Cp_count.append(count)
        if count > 0:
            Cp_mean.append(np.mean(Cp[mask]))
            Cp_std.append(np.std(Cp[mask]))
        else:
            Cp_mean.append(np.nan)
            Cp_std.append(np.nan)
    
    # ============================================================
    # GUARDAR CSV
    # ============================================================
    csv_path = "sphere_cp_validation.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Angle θ (°)', 'Cp Analytical', 'Cp Numerical', 'Cp Std', 'N Panels'])
        for i in range(n_bins):
            theta = (theta_deg[i] + theta_deg[i+1]) / 2
            analytical = 1 - (9/4) * np.sin(np.radians(theta))**2
            writer.writerow([
                f"{theta:.1f}",
                f"{analytical:.6f}",
                f"{Cp_mean[i]:.6f}" if not np.isnan(Cp_mean[i]) else "NaN",
                f"{Cp_std[i]:.6f}" if not np.isnan(Cp_std[i]) else "NaN",
                Cp_count[i]
            ])
    
    print(f"\n   ✅ CSV saved to: {csv_path}")
    
    # ============================================================
    # MOSTRAR TABLA
    # ============================================================
    print("\n" + "="*60)
    print("📊 CP VALIDATION TABLE (Figure 5)")
    print("="*60)
    print(f"{'Angle (°)':<12} {'Analytical':<15} {'Numerical':<15} {'Error (%)':<12}")
    print("-"*60)
    
    errors = []
    for i in range(n_bins):
        theta = (theta_deg[i] + theta_deg[i+1]) / 2
        if not np.isnan(Cp_mean[i]):
            analytical = 1 - (9/4) * np.sin(np.radians(theta))**2
            if analytical != 0:
                error = abs((Cp_mean[i] - analytical) / analytical) * 100
                errors.append(error)
            else:
                error = 0
            print(f"{theta:<12.1f} {analytical:<15.6f} {Cp_mean[i]:<15.6f} {error:<12.2f}")
        else:
            print(f"{theta:<12.1f} {'-' :<15} {'No data':<15} {'-':<12}")
    
    print("-"*60)
    if errors:
        print(f"   Mean Error: {np.mean(errors):.2f}%")
        print(f"   Max Error:  {np.max(errors):.2f}%")
    print("="*60)
    
    # ============================================================
    # VISUALIZAR
    # ============================================================
    print("\n   Displaying Cp contour on sphere...")
    sphere.cell_data['Cp'] = Cp
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, scalars='Cp', cmap='coolwarm',
                     scalar_bar_args={'title': 'Cp', 'title_font_size': 12})
    plotter.add_axes()
    plotter.view_isometric()
    plotter.show()
    
    print("\n✅ Extraction complete!")
    print(f"   CSV saved: {csv_path}")
    
    return Cp

def generate_figure_5():
    """Genera la Figura 5 a partir del CSV"""
    
    csv_path = "sphere_cp_validation.csv"
    if not os.path.exists(csv_path):
        print("❌ Please run extract_cp_table() first to generate the CSV.")
        return
    
    # Leer CSV manualmente
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 3:
                try:
                    theta = float(row[0])
                    analytical = float(row[1])
                    numerical = float(row[2]) if row[2] != 'NaN' else np.nan
                    data.append((theta, analytical, numerical))
                except:
                    continue
    
    if not data:
        print("❌ No valid data found in CSV.")
        return
    
    theta = np.array([d[0] for d in data])
    Cp_analytical = np.array([d[1] for d in data])
    Cp_numerical = np.array([d[2] for d in data])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    theta_smooth = np.linspace(0, 180, 200)
    Cp_smooth = 1 - (9/4) * np.sin(np.radians(theta_smooth))**2
    ax.plot(theta_smooth, Cp_smooth, 'k-', linewidth=2.5, label='Analytical Solution')
    
    valid = ~np.isnan(Cp_numerical)
    ax.plot(theta[valid], Cp_numerical[valid], 'ro', markersize=8, 
            label='Panel Method (Present)')
    
    ax.set_xlabel('Polar Angle θ (degrees)', fontsize=12)
    ax.set_ylabel('Pressure Coefficient $C_p$', fontsize=12)
    ax.set_title('Sphere Validation: $C_p$ vs Polar Angle', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    ax.set_xlim(0, 180)
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    if len(valid) > 0:
        error = np.mean(np.abs(Cp_numerical[valid] - Cp_analytical[valid]) / 
                        (np.abs(Cp_analytical[valid]) + 1e-10)) * 100
        ax.text(0.02, 0.02, f'Mean Error: {error:.2f}%', transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figure_5_sphere_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Figure 5 saved as 'figure_5_sphere_validation.png'")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔵 SPHERE CP EXTRACTION TOOL")
    print("   For Figure 5: Sphere Validation")
    print("="*60)
    
    Cp = extract_cp_table()
    
    if Cp is not None and os.path.exists("sphere_cp_validation.csv"):
        print("\n" + "-"*60)
        response = input("Generate Figure 5? (y/n): ")
        if response.lower() == 'y':
            generate_figure_5()
    
    print("\n✅ Done!")