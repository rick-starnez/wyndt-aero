# test_rotor_comparison.py (VERSIÓN FINAL - SIN CLAMP + CACHÉ)
"""
Compara rotor_sintetico.stl vs rotor_2_fixed.stl
- 12 puntos para diagnóstico rápido
- Reutiliza el solver (aprovecha el caché de la matriz de influencia)
- Usa torque crudo SIN clamp artificial
"""

import sys
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.panel_solver import PanelSolver3D


def analyze_geometry_symmetry(rotor, name):
    """
    Analiza la simetría del rotor para predecir comportamiento
    """
    print(f"\n🔍 ANALIZANDO SIMETRÍA: {name}")
    print("-"*50)
    
    centers = rotor.cell_centers().points
    normals = rotor.cell_data['Normals'] if 'Normals' in rotor.cell_data else None
    
    mesh_prop = rotor.compute_cell_sizes()
    areas = mesh_prop.cell_data['Area']
    area_total = np.sum(areas)
    
    print(f"   Paneles: {rotor.n_cells}")
    print(f"   Área total: {area_total:.6f} m²")
    
    # Distribución por cuadrantes
    centro_geom = np.mean(centers[:, :2], axis=0)
    cuadrantes = {'Q1':0, 'Q2':0, 'Q3':0, 'Q4':0}
    area_cuadrantes = {'Q1':0, 'Q2':0, 'Q3':0, 'Q4':0}
    
    for i, c in enumerate(centers):
        x = c[0] - centro_geom[0]
        y = c[1] - centro_geom[1]
        
        if x >= 0 and y >= 0:
            cuadrantes['Q1'] += 1
            area_cuadrantes['Q1'] += areas[i]
        elif x < 0 and y >= 0:
            cuadrantes['Q2'] += 1
            area_cuadrantes['Q2'] += areas[i]
        elif x < 0 and y < 0:
            cuadrantes['Q3'] += 1
            area_cuadrantes['Q3'] += areas[i]
        else:
            cuadrantes['Q4'] += 1
            area_cuadrantes['Q4'] += areas[i]
    
    print(f"\n   Distribución de área por cuadrante:")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        pct = area_cuadrantes[q] / area_total * 100
        print(f"      {q}: {area_cuadrantes[q]:.6f} m² ({pct:.1f}%)")
    
    # Área proyectada
    print(f"\n   Área proyectada en diferentes direcciones:")
    angulos_prueba = np.arange(0, 360, 30)
    
    if normals is not None:
        proj_areas = []
        for theta in angulos_prueba:
            rad = np.radians(theta)
            proj_dir = np.array([np.cos(rad), np.sin(rad), 0])
            proj = areas * np.abs(np.sum(normals * proj_dir, axis=1))
            proj_total = np.sum(proj)
            proj_areas.append(proj_total)
            print(f"      θ={theta:3d}°: {proj_total:.6f} m²")
        
        proj_std = np.std(proj_areas)
        proj_mean = np.mean(proj_areas)
        variacion_rel = proj_std / (proj_mean + 1e-10)
        
        print(f"\n   Variación del área proyectada: {variacion_rel*100:.2f}%")
        
        if variacion_rel < 0.05:
            print("   ⚠️ ¡ALERTA! El área proyectada es casi constante.")
            return {'es_simetrico': True, 'variacion': variacion_rel}
        else:
            print("   ✅ El área proyectada varía significativamente.")
            return {'es_simetrico': False, 'variacion': variacion_rel}
    
    return {'es_simetrico': None, 'variacion': 0}


def test_rotor_comparison():
    """Compara rotor_sintetico.stl vs rotor_2_fixed.stl"""
    
    print("\n" + "="*60)
    print("🚁 COMPARACIÓN DE ROTORES: SINTÉTICO vs FIXED")
    print("   (torque crudo SIN clamp, solver con caché)")
    print("="*60)
    
    rotores = [
        {"name": "Rotor Sintético (3 palas)", "file": "rotor_sintetico.stl", 
         "color": "blue", "marker": "o"},
        {"name": "Rotor 2 Fixed (4 palas)", "file": "rotor_2_fixed.stl", 
         "color": "red", "marker": "s"}
    ]
    
    resultados = {}
    angulos = np.linspace(0, 360, 12)  # 12 puntos para diagnóstico rápido
    
    # ============================================================
    # ANÁLISIS GEOMÉTRICO
    # ============================================================
    print("\n" + "="*60)
    print("📐 ANÁLISIS GEOMÉTRICO")
    print("="*60)
    
    for rotor_info in rotores:
        if not os.path.exists(rotor_info['file']):
            print(f"❌ No se encontró: {rotor_info['file']}")
            return
        
        rotor = pv.read(rotor_info['file'])
        sym_info = analyze_geometry_symmetry(rotor, rotor_info['name'])
        rotor_info['symmetry'] = sym_info
    
    # ============================================================
    # SIMULACIONES (con solver reutilizado)
    # ============================================================
    print("\n" + "="*60)
    print("💻 EJECUTANDO SIMULACIONES")
    print("="*60)
    
    for rotor_info in rotores:
        print(f"\n📁 Probando {rotor_info['name']}...")
        
        rotor = pv.read(rotor_info['file'])
        print(f"   Paneles: {rotor.n_cells}")
        
        # ============================================================
        # CLAVE: Crear el solver UNA SOLA VEZ por rotor
        # ============================================================
        solver = PanelSolver3D(rotor, scale=1.0, rho=1.225, verbose=False)
        
        Cl_values = []
        Cd_values = []
        torque_values = []          # Torque crudo (sin clamp)
        torque_corr_values = []     # Torque con la corrección *0.015 (por si quieres comparar)
        cp_values = []
        
        for i, theta in enumerate(angulos):
            print(f"   θ = {theta:6.1f}°  ({i+1}/{len(angulos)})", end='\r', flush=True)
            
            res = solver.solve(
                V_inf_magnitude=16.0,
                theta=theta,
                phi=90,
                use_gmres=(rotor.n_cells > 500)
            )
            
            # Valores CRUDOS del solver (sin corrección artificial)
            Cl_raw = res['forces']['Cl']
            Cd_raw = res['forces']['Cd']
            torque_raw = res['forces']['torque_z_real']  # ← Sin clamp
            
            # Guardar valores crudos
            Cl_values.append(Cl_raw)
            Cd_values.append(Cd_raw)
            torque_values.append(torque_raw)
            
            # Guardar también la versión "corregida" por si quieres comparar
            torque_corr_values.append(abs(torque_raw) * 0.015)
            
            cp_values.append(np.mean(res['Cp']))
        
        print()  # Salto de línea después del progreso
        
        resultados[rotor_info['name']] = {
            'theta': angulos,
            'Cl': np.array(Cl_values),
            'Cd': np.array(Cd_values),
            'torque': np.array(torque_values),
            'torque_corr': np.array(torque_corr_values),
            'Cp_mean': np.array(cp_values),
            'color': rotor_info['color'],
            'marker': rotor_info['marker'],
            'symmetry': rotor_info['symmetry']
        }
    
    # ============================================================
    # RESUMEN ESTADÍSTICO
    # ============================================================
    print("\n" + "="*60)
    print("📊 COMPARACIÓN DE RESULTADOS (TORQUE CRUDO)")
    print("="*60)
    
    for name, data in resultados.items():
        print(f"\n{name}:")
        print(f"   Cl:      max={np.max(data['Cl']):.6f}, min={np.min(data['Cl']):.6f}, "
              f"media={np.mean(data['Cl']):.6f}, std={np.std(data['Cl']):.6f}")
        print(f"   Cd:      max={np.max(data['Cd']):.6f}, min={np.min(data['Cd']):.6f}, "
              f"media={np.mean(data['Cd']):.6f}, std={np.std(data['Cd']):.6f}")
        print(f"   Torque:  max={np.max(data['torque']):.6f}, min={np.min(data['torque']):.6f}, "
              f"media={np.mean(data['torque']):.6f}, std={np.std(data['torque']):.6f}")
        
        torque_std = np.std(data['torque'])
        torque_mean = np.mean(data['torque'])
        
        if abs(torque_mean) > 1e-6:
            variacion_pct = (np.max(data['torque']) - np.min(data['torque'])) / abs(torque_mean) * 100
            print(f"   Variación de torque: {variacion_pct:.1f}%")
        
        if torque_std < 1e-6:
            print("   ❌ ¡TORQUE EXACTAMENTE CONSTANTE! Problema serio.")
        elif torque_std < 0.001:
            print("   ⚠️ Torque casi constante.")
        else:
            print(f"   ✅ Torque variable (std={torque_std:.6f})")
    
    # ============================================================
    # GRÁFICAS
    # ============================================================
    print("\n" + "="*60)
    print("📈 GENERANDO GRÁFICAS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparación de Rotores (torque crudo, sin clamp)', 
                 fontsize=16, fontweight='bold')
    
    # --- Cl vs θ ---
    ax1 = axes[0, 0]
    for name, data in resultados.items():
        ax1.plot(data['theta'], data['Cl'], 
                 color=data['color'], marker=data['marker'], linestyle='-',
                 linewidth=2, markersize=5, label=name, alpha=0.8)
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Ángulo θ (°)', fontsize=12)
    ax1.set_ylabel('Coeficiente de Sustentación (Cl)', fontsize=12)
    ax1.set_title('Cl vs Ángulo Azimutal', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim(0, 360)
    
    # --- Cd vs θ ---
    ax2 = axes[0, 1]
    for name, data in resultados.items():
        ax2.plot(data['theta'], data['Cd'], 
                 color=data['color'], marker=data['marker'], linestyle='-',
                 linewidth=2, markersize=5, label=name, alpha=0.8)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Ángulo θ (°)', fontsize=12)
    ax2.set_ylabel('Coeficiente de Arrastre (Cd)', fontsize=12)
    ax2.set_title('Cd vs Ángulo Azimutal', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_xlim(0, 360)
    
    # --- Torque CRUDO vs θ (LA CLAVE) ---
    ax3 = axes[1, 0]
    for name, data in resultados.items():
        ax3.plot(data['theta'], data['torque'], 
                 color=data['color'], marker=data['marker'], linestyle='-',
                 linewidth=2.5, markersize=6, label=name, alpha=0.8)
    ax3.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Ángulo θ (°)', fontsize=12)
    ax3.set_ylabel('Torque crudo (N·m)', fontsize=12)
    ax3.set_title('Torque CRUDO vs Ángulo (SIN clamp)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    ax3.set_xlim(0, 360)
    
    # --- Eficiencia L/D vs θ ---
    ax4 = axes[1, 1]
    for name, data in resultados.items():
        LD = data['Cl'] / (np.abs(data['Cd']) + 1e-10)
        ax4.plot(data['theta'], LD, 
                 color=data['color'], marker=data['marker'], linestyle='-',
                 linewidth=2, markersize=5, label=name, alpha=0.8)
    ax4.set_xlabel('Ángulo θ (°)', fontsize=12)
    ax4.set_ylabel('Eficiencia (L/D)', fontsize=12)
    ax4.set_title('Eficiencia vs Ángulo Azimutal', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')
    ax4.set_xlim(0, 360)
    
    plt.tight_layout()
    plt.savefig('rotor_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Gráfica guardada como: rotor_comparison.png")
    plt.show()
    
    return resultados


if __name__ == "__main__":
    test_rotor_comparison()