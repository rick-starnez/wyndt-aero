# generate_synthetic_rotor.py (VERSIÓN CON POSICIONES ASIMÉTRICAS)
"""
Generador de rotor sintético con malla de buena calidad para pruebas del solver

CAMBIO: Se añade el parámetro 'angles' a create_rotor() para permitir
posiciones NO uniformes de las palas (rompe simetría rotacional).
"""

import pyvista as pv
import numpy as np
import os


def create_airfoil_profile(chord, thickness, n_points=30):
    """
    Genera un perfil aerodinámico simple (simétrico) para la pala
    
    Args:
        chord: Cuerda del perfil (m)
        thickness: Espesor máximo del perfil (m)
        n_points: Número de puntos para discretizar el perfil
    
    Returns:
        x, y: Coordenadas del perfil
    """
    # Coordenadas a lo largo de la cuerda (0 a 1)
    x_frac = np.linspace(0, 1, n_points)
    
    # Perfil simétrico simple (similar a NACA)
    y_frac = 4 * thickness / chord * x_frac * (1 - x_frac)
    
    # Borde de ataque más redondeado
    y_frac = y_frac * np.sqrt(x_frac)
    
    # Escalar a la cuerda
    x = x_frac * chord
    y = y_frac
    
    return x, y


def create_blade(radius, chord, thickness, span, n_radial=20, n_chord=20):
    """
    Crea una pala individual extrudida en la dirección radial
    
    Args:
        radius: Radio del rotor (m)
        chord: Cuerda de la pala (m)
        thickness: Espesor máximo de la pala (m)
        span: Envergadura de la pala (m)
        n_radial: Número de puntos en dirección radial
        n_chord: Número de puntos en dirección de la cuerda
    
    Returns:
        pyvista.PolyData: Malla de la pala
    """
    
    # 1. Generar perfil base (en el plano XY)
    x_profile, y_profile = create_airfoil_profile(chord, thickness, n_chord)
    
    # 2. Crear puntos en 3D (extruir en Z)
    points = []
    y_positions = np.linspace(0, span, n_radial)
    scale_factors = 1.0 - 0.5 * (y_positions / span)
    
    for i, y_pos in enumerate(y_positions):
        scale = scale_factors[i]
        
        x_scaled = x_profile * scale
        z_scaled = y_profile * scale
        
        # Superficie superior
        for j in range(n_chord):
            points.append([x_scaled[j], y_pos, z_scaled[j]])
        
        # Superficie inferior
        for j in range(n_chord):
            points.append([x_scaled[j], y_pos, -z_scaled[j]])
    
    points = np.array(points)
    
    # 3. Crear las celdas (triángulos)
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
    
    # 4. Crear caras en los bordes
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


def create_rotor(radius=0.5, chord=0.15, thickness=0.03, 
                 num_blades=3, span=0.5, rotation_angle=0,
                 angles=None):
    """
    Crea un rotor completo con múltiples palas
    
    Args:
        radius: Radio del rotor (m)
        chord: Cuerda de la pala (m)
        thickness: Espesor máximo de la pala (m)
        num_blades: Número de palas
        span: Envergadura de la pala (m)
        rotation_angle: Ángulo de rotación inicial (grados)
        angles: Lista de ángulos personalizados para las palas.
                Si es None, usa distribución uniforme.
                Si se especifica, IGNORA num_blades y rotation_angle.
    
    Returns:
        pyvista.PolyData: Malla del rotor completo
    """
    
    print(f"\n🚁 Generando rotor sintético...")
    print(f"   Radio: {radius} m")
    print(f"   Cuerda: {chord} m")
    print(f"   Espesor: {thickness} m")
    print(f"   Envergadura: {span} m")
    
    # ============================================================
    # CAMBIO PRINCIPAL: usar ángulos personalizados si se especifican
    # ============================================================
    if angles is not None:
        angulos_palas = list(angles)
        num_blades = len(angulos_palas)
        print(f"   Modo: ÁNGULOS PERSONALIZADOS")
        print(f"   Ángulos: {angulos_palas}°")
        print(f"   Diferencia entre palas consecutivas: ", end='')
        diferencias = []
        for i in range(len(angulos_palas)):
            if i < len(angulos_palas) - 1:
                dif = angulos_palas[i+1] - angulos_palas[i]
            else:
                dif = 360 - angulos_palas[i] + angulos_palas[0]
            diferencias.append(f"{dif:.0f}°")
        print(", ".join(diferencias))
        print(f"   ⚠️  SIMETRÍA ROTA: las palas NO están equiespaciadas")
    else:
        angulos_palas = [rotation_angle + i * 360.0 / num_blades 
                         for i in range(num_blades)]
        print(f"   Modo: DISTRIBUCIÓN UNIFORME")
        print(f"   Número de palas: {num_blades}")
        print(f"   Ángulos: {[f'{a:.1f}°' for a in angulos_palas]}")
    
    # 1. Crear una pala
    blade = create_blade(radius, chord, thickness, span)
    print(f"   Pala creada: {blade.n_cells} celdas, {blade.n_points} puntos")
    
    # 2. Rotar y duplicar para crear todas las palas
    rotors = []
    for angulo in angulos_palas:
        blade_rotated = blade.rotate_z(angulo, point=(0, 0, 0))
        rotors.append(blade_rotated)
    
    # 3. Combinar todas las palas
    rotor_completo = rotors[0]
    for r in rotors[1:]:
        rotor_completo = rotor_completo + r
    
    # 4. Limpiar y unificar
    rotor_completo = rotor_completo.clean()
    
    # ============================================================
    # CORRECCIÓN: NORMALES HACIA AFUERA
    # ============================================================
    print("   🔧 Corrigiendo normales...")
    
    rotor_completo.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    
    centers = rotor_completo.cell_centers().points
    normals = rotor_completo.cell_data['Normals']
    
    dot_products = np.sum(normals * centers, axis=1)
    
    if np.mean(dot_products) < 0:
        print("   ⚠️ Normales apuntando hacia adentro. Invirtiendo...")
        rotor_completo.cell_data['Normals'] = -normals
        print("   ✅ Normales corregidas")
    else:
        print("   ✅ Normales ya apuntan hacia afuera")
    
    print(f"   Rotor completo: {rotor_completo.n_cells} celdas, {rotor_completo.n_points} puntos")
    
    return rotor_completo


def generate_and_save_rotor(radius=0.5, chord=0.15, thickness=0.03,
                           num_blades=3, span=0.5, rotation_angle=0,
                           filename="rotor_sintetico.stl",
                           angles=None):
    """
    Genera y guarda un rotor sintético
    """
    
    print("\n" + "="*60)
    print("🔧 GENERADOR DE ROTOR SINTÉTICO")
    print("="*60)
    
    rotor = create_rotor(radius, chord, thickness, num_blades, span, 
                        rotation_angle, angles=angles)
    
    rotor.save(filename)
    print(f"\n✅ Rotor guardado como: {filename}")
    
    print("\n📊 INFORMACIÓN DE LA MALLA:")
    print(f"   Archivo: {filename}")
    print(f"   Celdas: {rotor.n_cells}")
    print(f"   Puntos: {rotor.n_points}")
    
    if 'Normals' in rotor.cell_data:
        normals = rotor.cell_data['Normals']
        centers = rotor.cell_centers().points
        dot_avg = np.mean(np.sum(normals * centers, axis=1))
        print(f"   Producto punto promedio (normal · centro): {dot_avg:.4f}")
        if dot_avg > 0:
            print("   ✅ Normales apuntan hacia afuera")
        else:
            print("   ⚠️ Normales apuntan hacia adentro")
    
    return rotor


def generate_rotor_variants():
    """
    Genera varias variantes del rotor para pruebas
    """
    
    # Variante 1: Rotor pequeño (rápido)
    print("\n" + "="*60)
    print("🔄 GENERANDO VARIANTE 1: ROTOR PEQUEÑO")
    print("="*60)
    generate_and_save_rotor(
        radius=0.3,
        chord=0.08,
        thickness=0.015,
        num_blades=3,
        span=0.3,
        filename="rotor_pequeno.stl"
    )
    
    # Variante 2: Rotor mediano (recomendado)
    print("\n" + "="*60)
    print("🔄 GENERANDO VARIANTE 2: ROTOR MEDIANO")
    print("="*60)
    generate_and_save_rotor(
        radius=0.5,
        chord=0.15,
        thickness=0.03,
        num_blades=3,
        span=0.5,
        filename="rotor_mediano.stl"
    )
    
    # Variante 3: Rotor grande
    print("\n" + "="*60)
    print("🔄 GENERANDO VARIANTE 3: ROTOR GRANDE")
    print("="*60)
    generate_and_save_rotor(
        radius=0.8,
        chord=0.25,
        thickness=0.05,
        num_blades=3,
        span=0.8,
        filename="rotor_grande.stl"
    )


if __name__ == "__main__":
    # ============================================================
    # CAMBIO PRINCIPAL: usar ángulos ASIMÉTRICOS en lugar de uniformes
    # Original: [0°, 120°, 240°] → simetría rotacional perfecta
    # Nuevo:    [0°, 105°, 235°] → simetría rota
    # ============================================================
    generate_and_save_rotor(
        radius=0.5,
        chord=0.15,
        thickness=0.03,
        num_blades=3,
        span=0.5,
        filename="rotor_sintetico.stl",
        angles=[0, 105, 235]  # ← ÚNICO CAMBIO REAL
    )
    
    print("\n✅ ¡Rotores generados correctamente!")