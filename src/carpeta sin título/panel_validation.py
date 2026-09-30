# src/core/panel_validation.py
import numpy as np
from scipy.interpolate import interp1d
from typing import Dict

class NACAValidator:
    """Valida el solver contra datos experimentales NACA"""
    
    def __init__(self):
        # Datos de UIUC Airfoil Database (Re = 3e6)
        self.data = {
            'NACA0012': {
                'alpha': np.array([0, 2, 4, 6, 8, 10, 12, 14]),
                'Cl': np.array([0.0, 0.24, 0.48, 0.72, 0.94, 1.12, 1.24, 1.30]),
                'Cd': np.array([0.006, 0.007, 0.009, 0.013, 0.018, 0.026, 0.038, 0.055])
            }
        }
    
    def validate(self, airfoil: str, alphas: np.ndarray, 
                 Cl_sim: np.ndarray, Cd_sim: np.ndarray) -> Dict:
        """Compara simulación vs datos experimentales"""
        if airfoil not in self.data:
            raise ValueError(f"Perfil {airfoil} no disponible")
        
        exp = self.data[airfoil]
        
        Cl_exp_interp = interp1d(exp['alpha'], exp['Cl'], 
                                kind='cubic', bounds_error=False, 
                                fill_value='extrapolate')
        Cd_exp_interp = interp1d(exp['alpha'], exp['Cd'], 
                                kind='cubic', bounds_error=False,
                                fill_value='extrapolate')
        
        Cl_exp = Cl_exp_interp(alphas)
        Cd_exp = Cd_exp_interp(alphas)
        
        mask = (alphas >= exp['alpha'][0]) & (alphas <= exp['alpha'][-1])
        
        if np.sum(mask) == 0:
            return {'error': 'No hay ángulos en el rango experimental'}
        
        error_Cl = np.abs(Cl_sim[mask] - Cl_exp[mask]) / (np.max(Cl_exp) + 1e-6) * 100
        error_Cd = np.abs(Cd_sim[mask] - Cd_exp[mask]) / (np.max(Cd_exp) + 1e-6) * 100
        
        return {
            'Cl_exp': Cl_exp,
            'Cd_exp': Cd_exp,
            'Cl_sim': Cl_sim,
            'Cd_sim': Cd_sim,
            'Cl_error_mean': np.mean(error_Cl),
            'Cl_error_max': np.max(error_Cl),
            'Cd_error_mean': np.mean(error_Cd),
            'Cd_error_max': np.max(error_Cd)
        }
