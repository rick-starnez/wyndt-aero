# scale_rotor.py
import pyvista as pv
import os

# Ruta de tu STL original
stl_original = "/Users/maxx/Python/WindTunnel2/Veleta01a-Body.stl"

# Ruta de salida
stl_escalado = "/Users/maxx/Python/WindTunnel2/Veleta01a-Body_scaled.stl"

# Cargar
malla = pv.read(stl_original)
print(f"Original: {malla.n_cells} celdas, {malla.n_points} puntos")

# Escalar de mm a m (factor 0.001)
malla.points = malla.points * 0.001

# Guardar
malla.save(stl_escalado)
print(f"✅ Escalado guardado en: {stl_escalado}")

# Verificar dimensiones
bounds = malla.bounds
print(f"Dimensiones (m): {bounds[1]-bounds[0]:.3f} × {bounds[3]-bounds[2]:.3f} × {bounds[5]-bounds[4]:.3f}")