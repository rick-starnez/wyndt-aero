# generate_rotor_2.py (VERSIÓN DEFINITIVA - ROMPE SIMETRÍA)
"""
Genera un rotor con 4 palas en posiciones asimétricas
para que el torque varie con el ángulo
"""

import pyvista as pv
import numpy as np
import os


def create_blade_with_camber_and_twist(radius, chord, thickness, span, 
                                       n_radial=25, n_chord=25,
                                       camber=0.08, twist=5):
    """
    Crea una pala con combadura (camber) Y torsión (twist)
    para máxima asimetría aerodinámica
    """
    
    # 1. Perfil con combadura (línea media curva)
    x_frac = np.linspace(0, 1, n_chord)
    
    # Línea media (camber line) - curva parabólica
    camber_line = 4 * camber * chord * x_frac * (1 - x_frac)
    
    # Distribución de espesor (NACA simétrico)
    thickness_dist = 4 * thickness * x_frac * (1 - x_frac) * np.sqrt(x_frac)
    
    # Superficie superior e inferior
    x_profile = x_frac * chord
    y_upper = camber_line + thickness_dist / 2
    y_lower = camber_line - thickness_dist / 2
    
    # 2. Crear puntos 3D con torsión
    points = []
    y_positions = np.linspace(0, span, n_radial)
    
    # La cuerda se reduce hacia la punta (más realista)
    scale_factors = 1.0 - 0.3 * (y_positions / span)
    
    for i, y_pos in enumerate(y_positions):
        scale = scale_factors[i]
        
        # Torsión: máxima en la raíz, cero en la punta
        twist_rad = np.radians(twist * (1 - y_pos / span))
        cos_t = np.cos(twist_rad)
        sin_t = np.sin(twist_rad)
        
        # Escalar perfil
        x_scaled = x_profile * scale
        y_upper_scaled = y_upper * scale
        y_lower_scaled = y_lower * scale
        
        # Puntos superior (Z positivo) con torsión
        for j in range(n_chord):
            x_rot = x_scaled[j] * cos_t - y_upper_scaled[j] * sin_t
            z_rot = x_scaled[j] * sin_t + y_upper_scaled[j] * cos_t
            points.append([x_rot, y_pos, z_rot])
        
        # Puntos inferior (Z negativo) con torsión
        for j in range(n_chord):
            x_rot = x_scaled[j] * cos_t - y_lower_scaled[j] * sin_t
            z_rot = x_scaled[j] * sin_t + y_lower_scaled[j] * cos_t
            points.append([x_rot, y_pos, z_rot])
    
    points = np.array(points)
    
    # 3. Crear celdas (triángulos)
    cells = []
    n_points_per_section = 2 * n_chord
    
    for i in range(n_radial - 1):
        for j in range(n_chord - 1):
            idx_base = i * n_points_per_section
            
            # Superior
            cells.append([3, idx_base + j, idx_base + j + 1, idx_base + n_points_per_section + j])
            cells.append([3, idx_base + j + 1, idx_base + n_points_per_section + j + 1, idx_base + n_points_per_section + j])
            
            # Inferior
            idx_inf = idx_base + n_chord
            cells.append([3, idx_inf + j, idx_inf + n_points_per_section + j, idx_inf + j + 1])
            cells.append([3, idx_inf + j + 1, idx_inf + n_points_per_section + j, idx_inf + n_points_per_section + j + 1])
    
    # 4. Bordes (ataque y salida)
    for i in range(n_radial - 1):
        idx_base = i * n_points_per_section
        # Borde de ataque
        cells.append([3, idx_base, idx_base + n_points_per_section, idx_base + n_chord + n_points_per_section])
        cells.append([3, idx_base, idx_base + n_chord + n_points_per_section, idx_base + n_chord])
        # Borde de salida
        cells.append([3, idx_base + n_chord - 1, idx_base + n_points_per_section + n_chord - 1, idx_base + n_chord + n_points_per_section - 1])
        cells.append([3, idx_base + n_chord - 1, idx_base + n_chord + n_points_per_section - 1, idx_base + 2 * n_chord - 1])
    
    cells_flat = []
    for cell in cells:
        cells_flat.extend(cell)
    
    blade = pv.PolyData(points, np.array(cells_flat))
    return blade


def create_rotor_with_broken_symmetry(radius=0.4, chord=0.12, thickness=0.025,
                                      num_blades=4, span=0.4, camber=0.06, twist=5):
    """
    Crea un rotor con 4 palas en POSICIONES ASIMÉTRICAS
    para romper la simetría y generar torque variable
    """
    
    print(f"\n🚁 Generando rotor con simetría rota...")
    print(f"   Radio: {radius} m")
    print(f"   Cuerda: {chord} m")
    print(f"   Espesor: {thickness} m")
    print(f"   Número de palas: {num_blades}")
    print(f"   Combadura: {camber*100:.1f}%")
    print(f"   Torsión: {twist}°")
    
    # Crear una pala con combadura y torsión
    blade = create_blade_with_camber_and_twist(
        radius, chord, thickness, span,
        n_radial=25, n_chord=25,
        camber=camber, twist=twist
    )
    print(f"   Pala creada: {blade.n_cells} celdas")
    
    # ============================================================
    # CLAVE: POSICIONES ASIMÉTRICAS
    # No son 0°, 90°, 180°, 270°
    # Son posiciones OFFSET para romper la simetría de 4 ejes
    # ============================================================
    angulos_offset = [0, 85, 175, 265]  # ← ASIMÉTRICO (no son múltiplos de 90)
    
    rotors = []
    for offset in angulos_offset:
        blade_rotated = blade.rotate_z(offset, point=(0, 0, 0))
        rotors.append(blade_rotated)
    
    # Combinar
    rotor_completo = rotors[0]
    for r in rotors[1:]:
        rotor_completo = rotor_completo + r
    
    rotor_completo = rotor_completo.clean()
    
    # Corregir normales
    rotor_completo.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    centers = rotor_completo.cell_centers().points
    normals = rotor_completo.cell_data['Normals']
    if np.mean(np.sum(normals * centers, axis=1)) < 0:
        rotor_completo.cell_data['Normals'] = -normals
    
    print(f"   Rotor completo: {rotor_completo.n_cells} celdas")
    
    return rotor_completo


def generate_rotor_2_fixed():
    """Genera el Rotor 2 con simetría rota"""
    
    print("\n" + "="*60)
    print("🚁 GENERANDO ROTOR 2 CON SIMETRÍA ROTA")
    print("="*60)
    
    rotor = create_rotor_with_broken_symmetry(
        radius=0.4,
        chord=0.12,
        thickness=0.025,
        num_blades=4,
        span=0.4,
        camber=0.06,   # 6% de combadura
        twist=5        # 5° de torsión en la raíz
    )
    
    rotor.save("rotor_2_fixed.stl")
    print("\n✅ Rotor 2 guardado como: rotor_2_fixed.stl")
    
    # Verificar calidad
    mesh_prop = rotor.compute_cell_sizes()
    areas = mesh_prop.cell_data['Area']
    print(f"\n📊 CALIDAD DE LA MALLA:")
    print(f"   Paneles: {rotor.n_cells}")
    print(f"   Área promedio: {np.mean(areas):.8f} m²")
    print(f"   Área mínima: {np.min(areas):.8f} m²")
    print(f"   Área máxima: {np.max(areas):.8f} m²")
    
    # Verificar simetría (debería estar rota)
    centers = rotor.cell_centers().points
    centro = np.mean(centers[:, :2], axis=0)
    
    cuadrantes = {'Q1':0, 'Q2':0, 'Q3':0, 'Q4':0}
    for c in centers:
        x = c[0] - centro[0]
        y = c[1] - centro[1]
        if x >= 0 and y >= 0:
            cuadrantes['Q1'] += 1
        elif x < 0 and y >= 0:
            cuadrantes['Q2'] += 1
        elif x < 0 and y < 0:
            cuadrantes['Q3'] += 1
        else:
            cuadrantes['Q4'] += 1
    
    total = sum(cuadrantes.values())
    print(f"\n🔍 DISTRIBUCIÓN POR CUADRANTE (debería ser ASIMÉTRICA):")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        pct = cuadrantes[q] / total * 100
        print(f"   {q}: {cuadrantes[q]} puntos ({pct:.1f}%)")
    
    return rotor


if __name__ == "__main__":
    generate_rotor_2_fixed()