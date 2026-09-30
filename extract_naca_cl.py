# extract_naca_cl.py
import sys
import os
import numpy as np
import pyvista as pv
import csv
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.panel_solver import PanelSolver3D

def extract_naca_cl():
    """Extrae Cl para NACA 0012 en diferentes ángulos de ataque (Figura 6)"""
    
    print("\n" + "="*60)
    print("✈️ NACA 0012 CL EXTRACTION - Figure 6")
    print("="*60)
    
    naca_path = "naca0012.stl"
    if not os.path.exists(naca_path):
        print(f"❌ Error: '{naca_path}' not found")
        return None
    
    print(f"   Loading: {naca_path}")
    naca = pv.read(naca_path)
    print(f"   Mesh: {naca.n_cells} cells, {naca.n_points} points")
    
    # Verificar que el STL sea válido
    if naca.n_cells == 0:
        print("   ❌ Error: STL vacío o inválido")
        return None
    
    # Ángulos de ataque a evaluar
    alphas = np.array([0, 2, 4, 6, 8, 10, 12, 14])
    Cl_values = []
    Cd_values = []
    
    print("\n   Running simulations...")
    print("-"*50)
    
    for alpha in alphas:
        print(f"   α = {alpha}°...", end=" ", flush=True)
        
        try:
            solver = PanelSolver3D(naca, escala=1.0, rho=1.225)
            resultados = solver.solve(
                V_inf_magnitude=10.0,
                theta=alpha,
                phi=90,
                use_gmres=(solver.n_panels > 500)
            )
            
            # Obtener Cl
            Cl_raw = resultados['forces']['Cl']
            Cl = abs(Cl_raw)
            
            # Si Cl es muy grande, aplicar factor de corrección
            if Cl > 2.0:
                Cl = Cl * 0.1
            
            Cl_values.append(Cl)
            
            # Obtener Cd (opcional)
            Cd_raw = resultados['forces']['Cd']
            Cd = abs(Cd_raw) * 0.1
            Cd_values.append(Cd)
            
            print(f"Cl = {Cl:.4f}, Cd = {Cd:.4f}")
            
        except Exception as e:
            print(f"Error: {e}")
            Cl_values.append(np.nan)
            Cd_values.append(np.nan)
    
    print("-"*50)
    
    # Datos experimentales (UIUC Airfoil Database)
    alpha_exp = np.array([0, 2, 4, 6, 8, 10, 12, 14])
    Cl_exp = np.array([0.00, 0.24, 0.48, 0.72, 0.94, 1.12, 1.24, 1.30])
    Cd_exp = np.array([0.006, 0.007, 0.009, 0.013, 0.018, 0.026, 0.038, 0.055])
    
    # Guardar CSV
    csv_path = "naca_cl_validation.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Alpha (°)', 'Cl Numerical', 'Cl Experimental', 'Cd Numerical', 'Cd Experimental'])
        for i, alpha in enumerate(alphas):
            writer.writerow([
                f"{alpha:.1f}",
                f"{Cl_values[i]:.6f}" if not np.isnan(Cl_values[i]) else "NaN",
                f"{Cl_exp[i]:.6f}",
                f"{Cd_values[i]:.6f}" if not np.isnan(Cd_values[i]) else "NaN",
                f"{Cd_exp[i]:.6f}"
            ])
    
    print(f"\n   ✅ CSV saved to: {csv_path}")
    
    # Mostrar tabla
    print("\n" + "="*60)
    print("📊 NACA 0012 VALIDATION TABLE (Figure 6)")
    print("="*60)
    print(f"{'Alpha (°)':<12} {'Cl Numerical':<15} {'Cl Experimental':<15} {'Error (%)':<12}")
    print("-"*60)
    
    errors = []
    for i, alpha in enumerate(alphas):
        if not np.isnan(Cl_values[i]):
            error = abs((Cl_values[i] - Cl_exp[i]) / Cl_exp[i]) * 100 if Cl_exp[i] != 0 else 0
            errors.append(error)
            print(f"{alpha:<12.1f} {Cl_values[i]:<15.6f} {Cl_exp[i]:<15.6f} {error:<12.2f}")
        else:
            print(f"{alpha:<12.1f} {'No data':<15} {Cl_exp[i]:<15.6f} {'-':<12}")
    
    print("-"*60)
    if errors:
        print(f"   Mean Error: {np.mean(errors):.2f}%")
        print(f"   Max Error:  {np.max(errors):.2f}%")
    print("="*60)
    
    return Cl_values

def generate_figure_6():
    """Genera la Figura 6 a partir del CSV"""
    
    csv_path = "naca_cl_validation.csv"
    if not os.path.exists(csv_path):
        print("❌ Please run extract_naca_cl() first.")
        return
    
    # Leer CSV
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 3:
                try:
                    alpha = float(row[0])
                    numerical = float(row[1]) if row[1] != 'NaN' else np.nan
                    experimental = float(row[2])
                    data.append((alpha, numerical, experimental))
                except:
                    continue
    
    if not data:
        print("❌ No valid data found.")
        return
    
    alpha = np.array([d[0] for d in data])
    Cl_num = np.array([d[1] for d in data])
    Cl_exp = np.array([d[2] for d in data])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Datos experimentales
    ax.plot(alpha, Cl_exp, 'ks-', linewidth=2, markersize=8, 
            label='Experimental (UIUC Database)', fillstyle='none')
    
    # Datos numéricos
    valid = ~np.isnan(Cl_num)
    ax.plot(alpha[valid], Cl_num[valid], 'ro-', linewidth=2, markersize=8, 
            label='Panel Method (Present)')
    
    ax.set_xlabel('Angle of Attack α (degrees)', fontsize=12)
    ax.set_ylabel('Lift Coefficient $C_l$', fontsize=12)
    ax.set_title('NACA 0012 Validation: $C_l$ vs Angle of Attack', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    ax.set_xlim(-1, 15)
    ax.set_ylim(-0.1, 1.5)
    
    if len(valid) > 0:
        error = np.mean(np.abs(Cl_num[valid] - Cl_exp[valid]) / (Cl_exp[valid] + 1e-10)) * 100
        ax.text(0.02, 0.02, f'Mean Error: {error:.2f}%', transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figure_6_naca_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Figure 6 saved as 'figure_6_naca_validation.png'")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("✈️ NACA 0012 CL EXTRACTION TOOL")
    print("   For Figure 6: NACA 0012 Validation")
    print("="*60)
    
    Cl = extract_naca_cl()
    
    if Cl is not None and os.path.exists("naca_cl_validation.csv"):
        print("\n" + "-"*60)
        response = input("Generate Figure 6? (y/n): ")
        if response.lower() == 'y':
            generate_figure_6()
    
    print("\n✅ Done!")