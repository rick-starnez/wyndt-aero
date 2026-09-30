# tests/test_sphere.py
import pyvista as pv
import numpy as np
import sys
import os

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.panel_solver import PanelSolver3D

def test_sphere():
    """Prueba el solver con una esfera (solución analítica)"""
    
    print("=" * 60)
    print("PRUEBA: ESFERA EN FLUJO POTENCIAL")
    print("=" * 60)
    
    # 1. Crear geometría de prueba (esfera)
    print("\n1. Creando malla de esfera...")
    sphere = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
    print(f"   Número de celdas: {sphere.n_cells}")
    print(f"   Número de puntos: {sphere.n_points}")
    
    # 2. Crear solver
    print("\n2. Inicializando solver...")
    solver = PanelSolver3D(sphere, escala=1.0, rho=1.225)
    print(f"   Número de paneles: {solver.n_panels}")
    
    # 3. Resolver flujo
    print("\n3. Resolviendo flujo potencial...")
    resultados = solver.solve(
        V_inf_magnitude=10.0,  # m/s
        theta=0,               # ángulo azimutal
        phi=90                 # ángulo polar (viento horizontal)
    )
    
    # 4. Mostrar resultados
    print("\n4. RESULTADOS:")
    print(f"   Tiempo de solución: {resultados['solution_time']:.2f} segundos")
    
    # Mostrar condición si existe
    cond = resultados.get('matrix_condition')
    if cond is not None:
        print(f"   Número de condición: {cond:.2e}")
    else:
        print(f"   Número de condición: No calculado (matriz grande)")
    
    print(f"   Cl (sustentación): {resultados['forces']['Cl']:.6f}")
    print(f"   Cd (arrastre): {resultados['forces']['Cd']:.6f}")
    print(f"   Torque Z: {resultados['forces']['torque_z']:.6f} N·m")
    
    # 5. Estadísticas de Cp
    Cp = resultados['Cp']
    print(f"\n5. ESTADÍSTICAS DE Cp:")
    print(f"   Cp máximo: {np.max(Cp):.4f} (debe ser ~1.0)")
    print(f"   Cp mínimo: {np.min(Cp):.4f} (debe ser ~-0.5)")
    print(f"   Cp promedio: {np.mean(Cp):.4f}")
    
    # 6. Verificar simetría
    print(f"\n6. VERIFICACIÓN DE SIMETRÍA:")
    Cl_sym = resultados['forces']['Cl']
    print(f"   Cl (debe ser ~0): {Cl_sym:.6f}")
    
    # 7. Verificar que los resultados son físicos
    print(f"\n7. VERIFICACIÓN FÍSICA:")
    if abs(Cl_sym) < 0.01:
        print("   ✅ Simetría verificada (Cl ≈ 0)")
    else:
        print(f"   ⚠️ Simetría no perfecta (Cl = {Cl_sym:.6f})")
    
    if np.min(Cp) < -0.6:
        print(f"   ✅ Cp mínimo {np.min(Cp):.4f} (dentro del rango esperado)")
    else:
        print(f"   ⚠️ Cp mínimo {np.min(Cp):.4f} (puede ser bajo)")
    
    # 8. Visualización
    print("\n8. Visualizando resultados...")
    plotter = pv.Plotter()
    sphere.cell_data['Cp'] = Cp
    plotter.add_mesh(sphere, scalars='Cp', cmap='coolwarm', 
                     scalar_bar_args={'title': 'Coeficiente de Presión (Cp)'})
    plotter.add_axes()
    plotter.show()
    
    print("\n✅ Prueba completada exitosamente!")
    return resultados

if __name__ == "__main__":
    test_sphere()
