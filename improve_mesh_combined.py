# improve_mesh_gentle.py
import pyvista as pv
import numpy as np

def improve_mesh_gentle():
    """Mejora la malla sin cambiar la forma drásticamente"""
    
    print("\n" + "="*60)
    print("🛠️ MEJORANDO MALLA - ENFOQUE SUAVE")
    print("="*60)
    
    # 1. Cargar rotor original
    rotor = pv.read("Veleta01a-Body_scaled.stl")
    print(f"Paneles originales: {rotor.n_cells}")
    
    # 2. Solo suavizar (sin subdividir)
    rotor_smooth = rotor.smooth(n_iter=20, relaxation_factor=0.3)
    print(f"Paneles después de suavizar: {rotor_smooth.n_cells}")
    
    # 3. Guardar
    rotor_smooth.save("Veleta01a-Body_smooth.stl")
    print("✅ Rotor suavizado guardado")
    
    # 4. Verificar calidad
    mesh_prop = rotor_smooth.compute_cell_sizes()
    areas = mesh_prop.cell_data['Area']
    print(f"\n📊 CALIDAD:")
    print(f"   Área promedio: {np.mean(areas):.8f} m²")
    print(f"   Área mínima: {np.min(areas):.8f} m²")
    print(f"   Área máxima: {np.max(areas):.8f} m²")
    
    return rotor_smooth

if __name__ == "__main__":
    improve_mesh_gentle()