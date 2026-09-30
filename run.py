# run.py - Script principal para ejecutar el solver
import sys
import os
import pyvista as pv
import numpy as np

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.panel_solver import PanelSolver3D

def run_solver(stl_file, V_inf=10.0, theta=0, phi=90):
    """
    Ejecuta el solver en un archivo STL
    
    Args:
        stl_file: Ruta al archivo STL
        V_inf: Velocidad de corriente libre [m/s]
        theta: Ángulo azimutal [grados]
        phi: Ángulo polar [grados]
    """
    
    print("=" * 60)
    print("SOLVER DE PANELES 3D - TÚNEL DE VIENTO VIRTUAL")
    print("=" * 60)
    
    # 1. Cargar STL
    print(f"\n1. Cargando geometría: {stl_file}")
    if not os.path.exists(stl_file):
        print(f"❌ Error: Archivo {stl_file} no encontrado")
        return
    
    malla = pv.read(stl_file)
    print(f"   Celdas: {malla.n_cells}")
    print(f"   Puntos: {malla.n_points}")
    
    # 2. Crear solver
    print("\n2. Inicializando solver...")
    solver = PanelSolver3D(malla, escala=1.0, rho=1.225)
    print(f"   Paneles: {solver.n_panels}")
    
    # 3. Resolver
    print(f"\n3. Resolviendo flujo (V={V_inf} m/s, θ={theta}°, φ={phi}°)...")
    resultados = solver.solve(
        V_inf_magnitude=V_inf,
        theta=theta,
        phi=phi,
        use_gmres=(solver.n_panels > 1000)
    )
    
    # 4. Mostrar resultados
    print("\n4. RESULTADOS:")
    print(f"   Tiempo: {resultados['solution_time']:.2f} s")
    print(f"   Cl: {resultados['forces']['Cl']:.6f}")
    print(f"   Cd: {resultados['forces']['Cd']:.6f}")
    print(f"   Torque Z: {resultados['forces']['torque_z']:.6f} N·m")
    
    # 5. Visualizar
    print("\n5. Visualizando...")
    malla.cell_data['Cp'] = resultados['Cp']
    
    plotter = pv.Plotter()
    plotter.add_mesh(malla, scalars='Cp', cmap='coolwarm',
                     scalar_bar_args={'title': 'Coeficiente de Presión (Cp)'})
    plotter.add_axes()
    plotter.show()
    
    print("\n✅ Simulación completada!")

if __name__ == "__main__":
    # Ejemplo: generar una esfera y ejecutar
    print("Generando geometría de prueba (esfera)...")
    sphere = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
    
    # Guardar temporalmente
    temp_file = "temp_sphere.stl"
    sphere.save(temp_file)
    
    # Ejecutar solver
    run_solver(temp_file, V_inf=10.0, theta=0, phi=90)
    
    # Limpiar
    os.remove(temp_file)
