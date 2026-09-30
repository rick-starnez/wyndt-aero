# diagnostic_3_blades.py
"""
Diagnóstico profundo del rotor de 3 palas (Sintético)
Identifica por qué el torque es constante
"""

import sys
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.panel_solver import PanelSolver3D


def diagnose_geometry(rotor_file):
    """Analiza la geometría del rotor en detalle"""
    
    print("\n" + "="*60)
    print("📐 DIAGNÓSTICO DE GEOMETRÍA")
    print("="*60)
    
    rotor = pv.read(rotor_file)
    
    # 1. Información básica
    bounds = rotor.bounds
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]
    center = rotor.center
    
    print(f"\n📏 Dimensiones:")
    print(f"   X: {dx:.4f} m (rango: {bounds[0]:.4f} a {bounds[1]:.4f})")
    print(f"   Y: {dy:.4f} m (rango: {bounds[2]:.4f} a {bounds[3]:.4f})")
    print(f"   Z: {dz:.4f} m (rango: {bounds[4]:.4f} a {bounds[5]:.4f})")
    print(f"   Centro: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    
    # 2. Detectar orientación del disco del rotor
    print(f"\n🔍 Orientación del disco del rotor:")
    print(f"   Dimensión más pequeña: ", end='')
    dims = {'X': dx, 'Y': dy, 'Z': dz}
    min_dim = min(dims, key=dims.get)
    print(f"{min_dim} ({dims[min_dim]:.4f} m)")
    
    if min_dim == 'Z':
        print(f"   → El disco del rotor está en el plano XY")
        print(f"   → Las palas son coplanares con el disco (PROBLEMA POTENCIAL)")
    elif min_dim == 'Y':
        print(f"   → El disco del rotor está en el plano XZ")
    elif min_dim == 'X':
        print(f"   → El disco del rotor está en el plano YZ")
    
    # 3. Analizar orientación de las palas
    print(f"\n🎯 Análisis de palas individuales:")
    
    # Obtener centros de celdas
    centers = rotor.cell_centers().points
    
    # Calcular ángulo azimutal de cada celda (vista desde Z)
    angulos_azimutales = np.arctan2(centers[:, 1], centers[:, 0])
    angulos_grados = np.degrees(angulos_azimutales) % 360
    
    # Histograma de ángulos (debería mostrar picos en las palas)
    hist, bins = np.histogram(angulos_grados, bins=36, range=(0, 360))
    
    print(f"   Distribución angular de celdas (cada 10°):")
    for i in range(0, 36, 3):  # cada 30°
        bin_centro = (bins[i] + bins[i+1])/2
        count = hist[i]
        barra = "█" * (count // 20)
        print(f"      {bin_centro:5.0f}°: {count:5d} {barra}")
    
    # 4. Detectar número de palas (picos en el histograma)
    picos = []
    for i in range(len(hist)):
        if hist[i] > np.mean(hist) * 1.5:
            picos.append(bins[i])
    
    print(f"\n   Palas detectadas en: {[f'{p:.0f}°' for p in picos]}")
    
    # 5. Verificar si las palas son coplanares
    # Si todas las celdas tienen Z ≈ 0, son coplanares
    z_values = centers[:, 2]
    z_std = np.std(z_values)
    z_range = z_values.max() - z_values.min()
    
    print(f"\n   Variación en Z:")
    print(f"      std(Z) = {z_std:.6f} m")
    print(f"      rango(Z) = {z_range:.6f} m")
    
    if z_range < 0.01 * max(dx, dy):
        print(f"   ⚠️ ¡Las palas son COPLANARES con el disco!")
        print(f"      Esto causa que las fuerzas se cancelen simétricamente.")
    else:
        print(f"   ✅ Las palas tienen espesor en Z (no coplanares)")
    
    return {
        'dx': dx, 'dy': dy, 'dz': dz,
        'center': center,
        'min_dim': min_dim,
        'palas_en': picos,
        'z_std': z_std,
        'z_range': z_range,
        'coplanares': z_range < 0.01 * max(dx, dy)
    }


def diagnose_pressures(rotor_file, theta=0):
    """Analiza distribución de presiones en el rotor"""
    
    print("\n" + "="*60)
    print(f"💨 DIAGNÓSTICO DE PRESIONES (θ = {theta}°)")
    print("="*60)
    
    rotor = pv.read(rotor_file)
    solver = PanelSolver3D(rotor, scale=1.0, rho=1.225, verbose=False)
    res = solver.solve(V_inf_magnitude=16.0, theta=theta, phi=90)
    
    # 1. Cp por panel
    Cp = res['Cp']
    centers = solver.geometry.get_panel_centers()
    normals = solver.geometry.get_panel_normals()
    areas = solver.geometry.get_panel_areas()
    
    print(f"\n📊 Estadísticas de Cp:")
    print(f"   Cp max:  {np.max(Cp):.4f}")
    print(f"   Cp min:  {np.min(Cp):.4f}")
    print(f"   Cp mean: {np.mean(Cp):.4f}")
    print(f"   Cp std:  {np.std(Cp):.4f}")
    
    # 2. Fuerzas por panel
    p_gauge = solver.pressures - 101325.0
    panel_forces = -p_gauge[:, np.newaxis] * areas[:, np.newaxis] * normals
    
    # 3. Torque por panel (alrededor del eje Z)
    torque_panel = np.zeros(len(centers))
    for i in range(len(centers)):
        r_vec = np.array([centers[i, 0], centers[i, 1], 0])
        torque_vec = np.cross(r_vec, panel_forces[i])
        torque_panel[i] = torque_vec[2]
    
    print(f"\n⚙️ Torque por panel (eje Z):")
    print(f"   Torque max:  {np.max(torque_panel):.6f} N·m")
    print(f"   Torque min:  {np.min(torque_panel):.6f} N·m")
    print(f"   Torque sum:  {np.sum(torque_panel):.6f} N·m")
    
    # 4. Distribución angular del torque
    print(f"\n📐 Distribución angular del torque:")
    angulos = np.degrees(np.arctan2(centers[:, 1], centers[:, 0])) % 360
    
    # Agrupar torque por sector angular
    sectores = 12
    torque_por_sector = np.zeros(sectores)
    count_por_sector = np.zeros(sectores)
    
    for i, ang in enumerate(angulos):
        sector = int(ang / (360 / sectores)) % sectores
        torque_por_sector[sector] += torque_panel[i]
        count_por_sector[sector] += 1
    
    for s in range(sectores):
        ang_ini = s * 30
        ang_fin = (s + 1) * 30
        print(f"   Sector {ang_ini:3d}°-{ang_fin:3d}°: "
              f"torque = {torque_por_sector[s]:+.6f} N·m "
              f"({int(count_por_sector[s])} paneles)")
    
    # 5. Verificar cancelación
    torque_pos = np.sum(torque_panel[torque_panel > 0])
    torque_neg = np.sum(torque_panel[torque_panel < 0])
    
    print(f"\n🔄 Balance de torque:")
    print(f"   Contribución positiva: +{torque_pos:.6f} N·m")
    print(f"   Contribución negativa: {torque_neg:.6f} N·m")
    print(f"   Suma neta:             {torque_pos + torque_neg:.6f} N·m")
    
    if abs(torque_pos + torque_neg) < 0.01 * (abs(torque_pos) + abs(torque_neg)):
        print(f"   ⚠️ ¡CANCELACIÓN CASI PERFECTA!")
        print(f"      Las fuerzas positivas y negativas se anulan.")
    else:
        print(f"   ✅ Hay torque neto (no se cancela)")
    
    return {
        'Cp': Cp,
        'centers': centers,
        'normals': normals,
        'areas': areas,
        'torque_panel': torque_panel,
        'torque_total': np.sum(torque_panel),
        'torque_pos': torque_pos,
        'torque_neg': torque_neg
    }


def plot_diagnostic(diag_geom, diag_press, output_file='diagnostic_3_blades.png'):
    """Genera gráficas de diagnóstico"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Vista 3D de la geometría
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    rotor = pv.read('rotor_sintetico.stl')
    centers = rotor.cell_centers().points
    ax1.scatter(centers[::10, 0], centers[::10, 1], centers[::10, 2], 
                c='blue', s=1, alpha=0.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Vista 3D del Rotor')
    
    # 2. Distribución angular (vista superior)
    ax2 = fig.add_subplot(2, 3, 2)
    angulos = np.degrees(np.arctan2(centers[:, 1], centers[:, 0])) % 360
    ax2.hist(angulos, bins=72, color='blue', alpha=0.7)
    ax2.set_xlabel('Ángulo azimutal (°)')
    ax2.set_ylabel('Número de celdas')
    ax2.set_title('Distribución angular de celdas')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribución en Z
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(centers[:, 2], bins=50, color='green', alpha=0.7)
    ax3.set_xlabel('Z (m)')
    ax3.set_ylabel('Número de celdas')
    ax3.set_title('Distribución en Z (espesor)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Cp por panel (mapa)
    ax4 = fig.add_subplot(2, 3, 4)
    scatter = ax4.scatter(diag_press['centers'][:, 0], 
                          diag_press['centers'][:, 1],
                          c=diag_press['Cp'], cmap='coolwarm',
                          s=2, alpha=0.7)
    plt.colorbar(scatter, ax=ax4, label='Cp')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title(f'Distribución de Cp (θ={0}°)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    # 5. Torque por panel
    ax5 = fig.add_subplot(2, 3, 5)
    scatter = ax5.scatter(diag_press['centers'][:, 0], 
                          diag_press['centers'][:, 1],
                          c=diag_press['torque_panel'], cmap='RdBu_r',
                          s=2, alpha=0.7,
                          vmin=-np.max(np.abs(diag_press['torque_panel'])),
                          vmax=np.max(np.abs(diag_press['torque_panel'])))
    plt.colorbar(scatter, ax=ax5, label='Torque por panel (N·m)')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('Torque por panel')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    
    # 6. Torque por sector angular
    ax6 = fig.add_subplot(2, 3, 6)
    sectores = 12
    torque_sectores = []
    angulos_medios = []
    for s in range(sectores):
        ang_ini = s * (360/sectores)
        ang_fin = (s + 1) * (360/sectores)
        mask = (angulos >= ang_ini) & (angulos < ang_fin)
        torque_sectores.append(np.sum(diag_press['torque_panel'][mask]))
        angulos_medios.append((ang_ini + ang_fin) / 2)
    
    colors = ['red' if t > 0 else 'blue' for t in torque_sectores]
    ax6.bar(angulos_medios, torque_sectores, width=25, color=colors, alpha=0.7)
    ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_xlabel('Ángulo azimutal (°)')
    ax6.set_ylabel('Torque por sector (N·m)')
    ax6.set_title('Torque por sector angular')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Diagnóstico guardado en: {output_file}")
    plt.show()


def main():
    print("\n" + "="*60)
    print("🔬 DIAGNÓSTICO COMPLETO DEL ROTOR DE 3 PALAS")
    print("="*60)
    
    rotor_file = 'rotor_sintetico.stl'
    
    if not os.path.exists(rotor_file):
        print(f"❌ No se encontró: {rotor_file}")
        return
    
    # 1. Diagnóstico de geometría
    diag_geom = diagnose_geometry(rotor_file)
    
    # 2. Diagnóstico de presiones (en varios ángulos para comparar)
    diag_press_0 = diagnose_pressures(rotor_file, theta=0)
    diag_press_45 = diagnose_pressures(rotor_file, theta=45)
    diag_press_90 = diagnose_pressures(rotor_file, theta=90)
    
    # 3. Comparar torque en diferentes ángulos
    print("\n" + "="*60)
    print("📊 COMPARACIÓN DE TORQUE EN DIFERENTES ÁNGULOS")
    print("="*60)
    
    torque_0 = diag_press_0['torque_total']
    torque_45 = diag_press_45['torque_total']
    torque_90 = diag_press_90['torque_total']
    
    print(f"   θ =  0°: Torque = {torque_0:+.6f} N·m")
    print(f"   θ = 45°: Torque = {torque_45:+.6f} N·m")
    print(f"   θ = 90°: Torque = {torque_90:+.6f} N·m")
    
    std_manual = np.std([torque_0, torque_45, torque_90])
    print(f"\n   Desviación entre ángulos: {std_manual:.6f}")
    
    if std_manual < 0.01:
        print("   ❌ El torque no varía significativamente con el ángulo")
        print("      → Problema de simetría confirmado")
    else:
        print("   ✅ El torque varía con el ángulo")
    
    # 4. Gráficas de diagnóstico
    plot_diagnostic(diag_geom, diag_press_0)
    
    # 5. Conclusión
    print("\n" + "="*60)
    print("🎯 CONCLUSIÓN DEL DIAGNÓSTICO")
    print("="*60)
    
    if diag_geom['coplanares']:
        print("❌ PROBLEMA CONFIRMADO: Las palas son coplanares con el disco")
        print("   Las fuerzas de presión se cancelan simétricamente.")
        print("   Solución: Añadir torsión (twist) o combadura (camber) a las palas.")
    else:
        print("✅ Las palas NO son coplanares")
    
    if abs(diag_press_0['torque_pos'] + diag_press_0['torque_neg']) < 0.01 * abs(diag_press_0['torque_pos']):
        print("❌ PROBLEMA CONFIRMADO: Cancelación de torque casi perfecta")
        print(f"   Torque positivo: +{diag_press_0['torque_pos']:.4f} N·m")
        print(f"   Torque negativo: {diag_press_0['torque_neg']:.4f} N·m")
        print(f"   Suma: {diag_press_0['torque_total']:.6f} N·m")
        print("   La contribución de cada pala se cancela con las otras.")


if __name__ == "__main__":
    main()