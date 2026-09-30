# inspect_rotor_2.py
import pyvista as pv

rotor2 = pv.read("rotor_2.stl")
print("🔍 INSPECCIÓN DEL ROTOR 2:")
print(f"   Paneles: {rotor2.n_cells}")
print(f"   Puntos: {rotor2.n_points}")

# Verificar normales
rotor2.compute_normals(cell_normals=True, point_normals=False, inplace=True)
if 'Normals' in rotor2.cell_data:
    normals = rotor2.cell_data['Normals']
    print(f"   Normales calculadas: {normals.shape}")

# Visualizar
rotor2.plot(color='lightblue', show_edges=True)