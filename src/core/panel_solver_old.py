# src/core/panel_solver.py
import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import gmres
from typing import Dict
import time

from .panel_geometry import PanelGeometry
from .panel_utils import (
    build_influence_matrix, 
    compute_panel_velocities
)

class PanelSolver3D:
    """
    3D Panel Solver for incompressible potential flow
    """
    
    def __init__(self, malla_pv, escala=1.0, rho=1.225):
        self.geometry = PanelGeometry(malla_pv, escala)
        self.rho = rho
        self.V_inf = None
        self.V_inf_vec = None
        self.intensities = None
        self.velocities = None
        self.pressures = None
        self.Cp = None
        self.forces = None
        self.solution_time = 0.0
        self.n_panels = self.geometry.n_panels
        self.matrix_condition = None
        self.escala = escala
        
    def solve(self, V_inf_magnitude: float, theta: float, phi: float,
              use_gmres: bool = False, tol: float = 1e-6) -> Dict:
        
        start_time = time.time()
        
        self.V_inf = V_inf_magnitude
        rad_theta = np.radians(theta)
        rad_phi = np.radians(phi)
        
        self.V_inf_vec = np.array([
            V_inf_magnitude * np.sin(rad_phi) * np.cos(rad_theta),
            V_inf_magnitude * np.sin(rad_phi) * np.sin(rad_theta),
            V_inf_magnitude * np.cos(rad_phi)
        ], dtype=np.float64)
        
        centers = self.geometry.get_panel_centers().astype(np.float64)
        normals = self.geometry.get_panel_normals().astype(np.float64)
        areas = self.geometry.get_panel_areas().astype(np.float64)
        
        print(f"Building influence matrix ({self.n_panels}x{self.n_panels})...")
        A = build_influence_matrix(centers, normals, areas)
        
        if self.n_panels < 500:
            cond = np.linalg.cond(A)
            self.matrix_condition = cond
            print(f"Condition number: {cond:.2e}")
        
        b = -np.dot(self.V_inf_vec, normals.T)
        
        print("Solving linear system...")
        if use_gmres or self.n_panels > 1000:
            try:
                intensities, info = gmres(A, b, rtol=tol, maxiter=1000)
            except TypeError:
                intensities, info = gmres(A, b, tol=tol, maxiter=1000)
            
            if info != 0:
                print(f"GMRES did not converge (info={info}), using direct solver")
                intensities = solve(A, b)
        else:
            intensities = solve(A, b)
        
        self.intensities = intensities.astype(np.float64)
        
        print("Computing panel velocities...")
        self.velocities = compute_panel_velocities(
            centers, normals, areas, intensities, self.V_inf_vec
        )
        
        self._compute_pressures()
        self._compute_forces_corrected(centers, normals, areas)
        
        self.solution_time = time.time() - start_time
        print(f"Solution completed in {self.solution_time:.2f} seconds")
        
        return self.get_results()
    
    def _compute_pressures(self):
        V_mag = np.linalg.norm(self.velocities, axis=1)
        V_inf_mag = self.V_inf
        
        self.Cp = 1.0 - (V_mag / V_inf_mag) ** 2
        p_inf = 101325.0
        q_inf = 0.5 * self.rho * V_inf_mag**2
        self.pressures = p_inf + q_inf * self.Cp
    
    def _compute_forces_corrected(self, centers, normals, areas):
        
        # ============================================================
        # CORRECCIÓN: Verificar y corregir áreas
        # ============================================================
        avg_area = np.mean(areas)
        print(f"\n   📐 Área promedio recibida: {avg_area:.8f} m²")
        
        if avg_area > 0.01:
            print(f"   ⚠️ Aplicando corrección de escala (factor 1e-6)...")
            areas = areas * 1e-6
            print(f"   ✅ Nueva área promedio: {np.mean(areas):.8f} m²")
        else:
            print(f"   ✅ Áreas en rango correcto")
        
        # ============================================================
        # 1. FIND ROTATION CENTER
        # ============================================================
        center_xy = np.mean(centers[:, :2], axis=0)
        rotation_center = np.array([center_xy[0], center_xy[1], 0.0])
        
        print(f"\n🔧 ROTATION CENTER:")
        print(f"   ({rotation_center[0]:.4f}, {rotation_center[1]:.4f}, {rotation_center[2]:.4f}) m")
        
        # ============================================================
        # 2. CORRECT NORMALS
        # ============================================================
        normals_corrected = normals.copy()
        for i in range(self.n_panels):
            r_vec = centers[i] - rotation_center
            r_xy = np.array([r_vec[0], r_vec[1], 0.0])
            if np.linalg.norm(r_xy) > 1e-10:
                if np.dot(normals_corrected[i], r_xy) < 0:
                    normals_corrected[i] = -normals_corrected[i]
        
        # ============================================================
        # 3. COMPUTE FORCES
        # ============================================================
        p_gauge = self.pressures - 101325.0
        
        panel_forces = np.zeros((self.n_panels, 3))
        for i in range(self.n_panels):
            panel_forces[i] = p_gauge[i] * areas[i] * normals_corrected[i]
        
        F_total = np.sum(panel_forces, axis=0)
        
        # ============================================================
        # 4. LIFT AND DRAG (IN X-Y PLANE)
        # ============================================================
        V_dir_xy = np.array([self.V_inf_vec[0], self.V_inf_vec[1], 0.0])
        v_mag_xy = np.linalg.norm(V_dir_xy)
        if v_mag_xy > 0:
            V_dir_xy = V_dir_xy / v_mag_xy
        else:
            V_dir_xy = np.array([1.0, 0.0, 0.0])
        
        drag = np.dot(F_total[:2], V_dir_xy[:2])
        lift_vector = F_total[:2] - drag * V_dir_xy[:2]
        lift = np.linalg.norm(lift_vector)
        
        # ============================================================
        # 5. DIMENSIONLESS COEFFICIENTS (CON DETECCIÓN MEJORADA)
        # ============================================================
        bbox = self.geometry.get_bounding_box()
        dx = bbox[1] - bbox[0]
        dy = bbox[3] - bbox[2]
        dz = bbox[5] - bbox[4]
        
        # ============================================================
        # 5a. DETECCIÓN MEJORADA: ROTOR vs PERFIL_2D
        # ============================================================
        # Analizar distribución de puntos en X-Y para detectar rotor
        r = np.sqrt(centers[:, 0]**2 + centers[:, 1]**2)
        
        # Contar puntos en cada cuadrante
        q1 = np.sum((centers[:, 0] > 0) & (centers[:, 1] > 0))
        q2 = np.sum((centers[:, 0] < 0) & (centers[:, 1] > 0))
        q3 = np.sum((centers[:, 0] < 0) & (centers[:, 1] < 0))
        q4 = np.sum((centers[:, 0] > 0) & (centers[:, 1] < 0))
        cuadrantes = [q1, q2, q3, q4]
        
        # Si la distribución es equilibrada (todos los cuadrantes tienen puntos), es un rotor
        if min(cuadrantes) > 0 and max(cuadrantes) / min(cuadrantes) < 5:
            # Es un rotor
            radius = np.max(r)
            ref_area = np.pi * radius**2
            geometria_tipo = "ROTOR"
            print(f"   📐 Geometría detectada: ROTOR (distribución en {len([c for c in cuadrantes if c > 0])} cuadrantes)")
        else:
            # Es un perfil 2D
            ref_area = dx * dz
            geometria_tipo = "PERFIL_2D"
            print(f"   📐 Geometría detectada: PERFIL_2D")
        
        print(f"   Área de referencia: {ref_area:.6f} m²")
        
        q_inf = 0.5 * self.rho * self.V_inf**2
        
        if q_inf * ref_area > 0:
            Cl = lift / (q_inf * ref_area)
            Cd = drag / (q_inf * ref_area)
        else:
            Cl = 0.0
            Cd = 0.0
        
        # ============================================================
        # 6. TORQUE (Z-COMPONENT ONLY)
        # ============================================================
        torque_z = 0.0
        for i in range(self.n_panels):
            r_vec = centers[i] - rotation_center
            r_xy = np.array([r_vec[0], r_vec[1], 0.0])
            F_xy = np.array([panel_forces[i][0], panel_forces[i][1], 0.0])
            torque_z += np.cross(r_xy, F_xy)[2]
        
        # ============================================================
        # 7. SAVE RESULTS
        # ============================================================
        self.forces = {
            'panel': panel_forces,
            'total': F_total,
            'drag': drag,
            'lift': lift,
            'lift_vector': lift_vector,
            'Cl': Cl,
            'Cd': Cd,
            'torque_z': torque_z,
            'rotation_center': rotation_center,
            'ref_area': ref_area,
            'geometria_tipo': geometria_tipo,
            'dimensions': {'dx': dx, 'dy': dy, 'dz': dz}
        }
        
        # ============================================================
        # 8. DEBUG INFORMATION
        # ============================================================
        print(f"\n🔧 FORCE RESULTS:")
        print(f"   Dimensiones: {dx:.4f} × {dy:.4f} × {dz:.4f} m")
        print(f"   Tipo: {geometria_tipo}")
        print(f"   Reference area: {ref_area:.6f} m²")
        print(f"   Total force X-Y: ({F_total[0]:.2f}, {F_total[1]:.2f}) N")
        print(f"   Lift (L): {lift:.2f} N")
        print(f"   Drag (D): {drag:.2f} N")
        print(f"   Lift coefficient Cl: {Cl:.6f}")
        print(f"   Drag coefficient Cd: {Cd:.6f}")
        print(f"   Torque Z: {torque_z:.6f} N·m")
        if abs(drag) > 0:
            print(f"   L/D Efficiency: {abs(lift/drag):.2f}")
        else:
            print(f"   L/D Efficiency: ∞")
    
    def get_results(self) -> Dict:
        return {
            'intensities': self.intensities,
            'velocities': self.velocities,
            'pressures': self.pressures,
            'Cp': self.Cp,
            'forces': self.forces,
            'solution_time': self.solution_time,
            'n_panels': self.n_panels,
            'matrix_condition': self.matrix_condition
        }
    
    def get_field_velocity(self, points: np.ndarray) -> np.ndarray:
        centers = self.geometry.get_panel_centers().astype(np.float64)
        normals = self.geometry.get_panel_normals().astype(np.float64)
        areas = self.geometry.get_panel_areas().astype(np.float64)
        intensities = self.intensities.astype(np.float64)
        
        V_field = np.zeros_like(points, dtype=np.float64)
        
        for i, point in enumerate(points):
            V = self.V_inf_vec.copy()
            for j in range(self.n_panels):
                r = point - centers[j]
                dist = np.linalg.norm(r)
                if dist < 1e-10:
                    continue
                factor = intensities[j] * areas[j] / (4.0 * np.pi * dist**3)
                V += factor * r
            V_field[i] = V
        
        return V_field