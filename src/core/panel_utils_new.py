# src/core/panel_utils_new.py
import numpy as np
from numba import jit, prange

@jit(nopython=True, cache=True)
def panel_influence(point, panel_center, panel_normal, panel_area):
    """
    Calcula la influencia de un panel en un punto.
    Para una fuente constante: A_ij = 1/(4π) * ∫∫ (P-Q)·n / |P-Q|³ dS
    """
    r = point - panel_center
    dist = np.linalg.norm(r)
    
    if dist < 1e-12:
        return 0.5  # Auto-influencia
    
    dot = np.dot(r, panel_normal)
    influence = dot / (4.0 * np.pi * dist**3) * panel_area
    
    return influence

@jit(nopython=True, parallel=True, cache=True)
def build_influence_matrix(centers, normals, areas):
    """Construye la matriz de influencia en paralelo."""
    n = len(centers)
    A = np.zeros((n, n))
    
    for i in prange(n):
        for j in range(n):
            if i == j:
                A[i, j] = 0.5
            else:
                A[i, j] = panel_influence(centers[i], centers[j], normals[j], areas[j])
    
    return A

@jit(nopython=True, cache=True)
def compute_induced_velocity(point, centers, normals, areas, intensities):
    """
    Calcula la velocidad inducida por todos los paneles en un punto.
    V = Σ σⱼ * Aⱼ * (P - Qⱼ) / (4π * |P - Qⱼ|³)
    """
    V = np.zeros(3)
    
    for j in range(len(centers)):
        r = point - centers[j]
        dist = np.linalg.norm(r)
        
        if dist < 1e-12:
            continue
        
        factor = intensities[j] * areas[j] / (4.0 * np.pi * dist**3)
        V += factor * r
    
    return V

@jit(nopython=True, parallel=True, cache=True)
def compute_panel_velocities(centers, normals, areas, intensities, V_inf):
    """Calcula la velocidad en cada panel."""
    n = len(centers)
    velocities = np.zeros((n, 3))
    
    for i in prange(n):
        V = V_inf.copy()
        V += compute_induced_velocity(centers[i], centers, normals, areas, intensities)
        velocities[i] = V
    
    return velocities