# src/core/panel_solver.py (VERSIÓN MEJORADA)
import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import gmres
import time

from .panel_geometry import PanelGeometry
from .panel_utils_new import build_influence_matrix, compute_panel_velocities


class PanelSolver3D:
    """
    Solver de paneles 3D.
    
    Novedades:
    - Cachea la matriz de influencia para no reconstruirla en cada ángulo
    - Expone 'torque_z_real' (sin clamp) además de 'torque_z'
    """
    
    def __init__(self, mesh, scale=1.0, rho=1.225, verbose=True, **kwargs):
        if 'escala' in kwargs:
            scale = kwargs['escala']
        
        self.geometry = PanelGeometry(mesh, scale)
        self.rho = rho
        self.verbose = verbose
        self.n_panels = self.geometry.n_panels
        self.V_inf = None
        self.V_inf_vec = None
        self.intensities = None
        self.velocities = None
        self.pressures = None
        self.Cp = None
        self.forces = None
        self.solution_time = 0.0
        
        # ============================================================
        # CACHE DE LA MATRIZ DE INFLUENCIA
        # ============================================================
        self._A_cached = None
        self._centers_cached = None
        self._normals_cached = None
        self._areas_cached = None
    
    def _get_influence_data(self):
        """
        Devuelve (A, centers, normals, areas).
        Si ya están cacheados, los reutiliza.
        """
        if self._A_cached is None:
            if self.verbose:
                print("   🔧 Construyendo matriz de influencia (primera vez)...")
            
            self._centers_cached = self.geometry.get_panel_centers().astype(np.float64)
            self._normals_cached = self.geometry.get_panel_normals().astype(np.float64)
            self._areas_cached = self.geometry.get_panel_areas().astype(np.float64)
            
            self._A_cached = build_influence_matrix(
                self._centers_cached, 
                self._normals_cached, 
                self._areas_cached
            )
        else:
            if self.verbose:
                print("   ♻️ Reutilizando matriz de influencia cacheada")
        
        return (self._A_cached, self._centers_cached, 
                self._normals_cached, self._areas_cached)
        
    def solve(self, V_inf_magnitude: float, theta: float, phi: float,
              use_gmres: bool = False, tol: float = 1e-6):
        """
        Resuelve el flujo potencial.
        """
        start_time = time.time()
        
        # 1. Vector de corriente libre
        self.V_inf = V_inf_magnitude
        rad_theta = np.radians(theta)
        rad_phi = np.radians(phi)
        self.V_inf_vec = np.array([
            V_inf_magnitude * np.sin(rad_phi) * np.cos(rad_theta),
            V_inf_magnitude * np.sin(rad_phi) * np.sin(rad_theta),
            V_inf_magnitude * np.cos(rad_phi)
        ], dtype=np.float64)
        
        if self.verbose:
            print(f"\n🔧 Resolviendo flujo potencial:")
            print(f"   V∞ = {V_inf_magnitude:.2f} m/s, θ = {theta:.1f}°, φ = {phi:.1f}°")
            print(f"   Paneles: {self.n_panels}")
        
        # 2. Obtener datos (con cache)
        A, centers, normals, areas = self._get_influence_data()
        
        # 3. Vector RHS (esto SÍ cambia con theta/phi)
        b = -np.dot(self.V_inf_vec, normals.T)
        
        # 4. Resolver sistema lineal
        if self.verbose:
            print("   Resolviendo sistema lineal...")
        
        if use_gmres and self.n_panels > 500:
            intensities, info = gmres(A, b, rtol=tol, maxiter=1000)
            if info != 0:
                if self.verbose:
                    print(f"   ⚠️ GMRES no convergió (info={info}), usando solver directo")
                intensities = solve(A, b)
        else:
            intensities = solve(A, b)
        
        self.intensities = intensities.astype(np.float64)
        
        # 5. Calcular velocidades
        if self.verbose:
            print("   Calculando velocidades...")
        self.velocities = compute_panel_velocities(
            centers, normals, areas, intensities, self.V_inf_vec
        )
        
        # 6. Presiones
        self._compute_pressures()
        
        # 7. Fuerzas y torques
        self._compute_forces(centers, normals, areas)
        
        # 8. Tiempo
        self.solution_time = time.time() - start_time
        if self.verbose:
            print(f"   ✅ Solución completada en {self.solution_time:.2f} s")
        
        return self.get_results()
    
    def _compute_pressures(self):
        V_mag = np.linalg.norm(self.velocities, axis=1)
        V_inf_mag = self.V_inf
        self.Cp = 1.0 - (V_mag / V_inf_mag) ** 2
        q_inf = 0.5 * self.rho * V_inf_mag**2
        p_inf = 101325.0
        self.pressures = p_inf + q_inf * self.Cp
    
    def _compute_forces(self, centers, normals, areas):
        p_gauge = self.pressures - 101325.0
        
        panel_forces = np.zeros((self.n_panels, 3))
        for i in range(self.n_panels):
            panel_forces[i] = -p_gauge[i] * areas[i] * normals[i]
        
        F_total = np.sum(panel_forces, axis=0)
        
        V_dir = self.V_inf_vec / (self.V_inf + 1e-10)
        drag = np.dot(F_total, V_dir)
        lift_vector = F_total - drag * V_dir
        lift = np.linalg.norm(lift_vector)
        
        bbox = self.geometry.get_bounding_box()
        dx = bbox[1] - bbox[0]
        dy = bbox[3] - bbox[2]
        dz = bbox[5] - bbox[4]
        
        if dz > 0.3 * dx:
            ref_area = dx * dz
            geometria_tipo = "PERFIL_2D"
        else:
            r = np.sqrt(centers[:, 0]**2 + centers[:, 1]**2)
            radius = np.max(r)
            ref_area = np.pi * radius**2
            geometria_tipo = "ROTOR"
        
        q_inf = 0.5 * self.rho * self.V_inf**2
        Cl = lift / (q_inf * ref_area) if q_inf * ref_area > 0 else 0.0
        Cd = drag / (q_inf * ref_area) if q_inf * ref_area > 0 else 0.0
        
        rotation_center = np.mean(centers[:, :2], axis=0)
        rotation_center = np.array([rotation_center[0], rotation_center[1], 0.0])
        
        torque = np.zeros(3)
        for i in range(self.n_panels):
            r_vec = centers[i] - rotation_center
            torque += np.cross(r_vec, panel_forces[i])
        
        # ============================================================
        # RESULTADOS: torque_z SIN clamp
        # ============================================================
        self.forces = {
            'panel': panel_forces,
            'total': F_total,
            'drag': drag,
            'lift': lift,
            'lift_vector': lift_vector,
            'Cl': Cl,
            'Cd': Cd,
            'torque': torque,
            'torque_z': torque[2],           # Crudo
            'torque_z_real': torque[2],      # ← NUEVO alias
            'rotation_center': rotation_center,
            'ref_area': ref_area,
            'geometria_tipo': geometria_tipo,
            'dimensions': {'dx': dx, 'dy': dy, 'dz': dz}
        }
        
        if self.verbose:
            print(f"\n🔧 RESULTADOS:")
            print(f"   Tipo: {geometria_tipo}")
            print(f"   Cl: {Cl:.6f}")
            print(f"   Cd: {Cd:.6f}")
            print(f"   Torque Z (crudo): {torque[2]:.6f} N·m")
    
    def get_results(self):
        return {
            'intensities': self.intensities,
            'velocities': self.velocities,
            'pressures': self.pressures,
            'Cp': self.Cp,
            'forces': self.forces,
            'solution_time': self.solution_time,
            'n_panels': self.n_panels
        }
    
    def get_field_velocity(self, points):
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