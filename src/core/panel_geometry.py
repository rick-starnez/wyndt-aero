# src/core/panel_geometry.py
import numpy as np
import pyvista as pv
from typing import List, Dict, Tuple

class PanelGeometry:
    """Manages triangular panel geometry with quality checks"""
    
    def __init__(self, malla_pv, escala=1.0):
        """
        Args:
            malla_pv: PyVista mesh (STL)
            escala: Scale factor to convert to meters
        """
        self.original_mesh = malla_pv
        self.scale = escala
        self.panels = []
        self.quality_metrics = {}
        
        # ============================================================
        # 1. VERIFICAR Y REPARAR GEOMETRÍA
        # ============================================================
        self._validate_and_repair_mesh()
        
        # ============================================================
        # 2. EXTRAER PANELES
        # ============================================================
        self._extract_panels()
        
        # ============================================================
        # 3. VERIFICAR CALIDAD DE PANELES
        # ============================================================
        self._validate_panels()
    
    def _validate_and_repair_mesh(self):
        """Valida y repara la malla antes de extraer paneles"""
        print("\n🔧 VALIDANDO GEOMETRÍA")
        print("-"*40)
        
        # 1. Verificar que sea un objeto PolyData
        if not isinstance(self.original_mesh, pv.PolyData):
            raise ValueError("La malla debe ser de tipo PolyData")
        
        # 2. Verificar que tenga caras
        if self.original_mesh.n_cells == 0:
            raise ValueError("La malla no tiene celdas")
        
        # 3. Verificar que sea todo triángulos
        if not self.original_mesh.is_all_triangles:
            print("   ⚠️ La malla no es completamente triangular. Triangulando...")
            self.original_mesh = self.original_mesh.triangulate()
        
        # 4. Verificar normales
        print("   Verificando normales...")
        self.original_mesh.compute_normals(
            cell_normals=True, 
            point_normals=False, 
            inplace=True,
            flip_normals=False
        )
        
        # 5. Verificar si es watertight (sin bordes libres)
        # Para PyVista, usamos extract_edges para detectar bordes libres
        try:
            edges = self.original_mesh.extract_edges()
            n_free_edges = edges.n_cells
            if n_free_edges > 0:
                print(f"   ⚠️ La malla tiene {n_free_edges} bordes libres")
                print("   Intentando reparar...")
                # Intentar reparar con fill_holes
                try:
                    self.original_mesh = self.original_mesh.fill_holes(hole_size=10.0)
                    print("   ✅ Reparación completada")
                except:
                    print("   ⚠️ No se pudo reparar automáticamente")
            else:
                print("   ✅ Malla watertight (sin bordes libres)")
        except:
            print("   ⚠️ No se pudo verificar si es watertight")
        
        # 6. Limpiar geometría (eliminar duplicados, etc.)
        print("   Limpiando geometría...")
        self.original_mesh = self.original_mesh.clean()
        print(f"   ✅ Malla limpia: {self.original_mesh.n_cells} celdas")
        
        print("-"*40)
    
    def _extract_panels(self):
        """Extract triangular panels from mesh with correct scaling"""
        # Escalar los puntos a metros
        points = self.original_mesh.points * self.scale
        
        if hasattr(self.original_mesh, 'faces'):
            faces = self.original_mesh.faces.reshape(-1, 4)[:, 1:]
        else:
            faces = self.original_mesh.cells.reshape(-1, 4)[:, 1:]
        
        self.panels = []
        self.quality_metrics = {
            'areas': [],
            'aspect_ratios': [],
            'skewness': []
        }
        
        for idx, face in enumerate(faces):
            v0 = points[face[0]]
            v1 = points[face[1]]
            v2 = points[face[2]]
            
            # Centro del panel
            center = (v0 + v1 + v2) / 3.0
            
            # Vectores para normal y área (EN METROS)
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)
            
            # ============================================================
            # CALIDAD DEL PANEL: Aspect Ratio y Skewness
            # ============================================================
            # Aspect Ratio: relación entre el lado más largo y el más corto
            lengths = [np.linalg.norm(edge1), np.linalg.norm(edge2), 
                      np.linalg.norm(v2 - v1)]
            aspect_ratio = max(lengths) / (min(lengths) + 1e-12)
            
            # Skewness: qué tan cerca está de ser equilátero
            # 0 = equilátero, 1 = degenerado
            avg_length = np.mean(lengths)
            if avg_length > 0:
                skewness = np.std(lengths) / avg_length
            else:
                skewness = 1.0
            
            # Normalizar (asegurar que apunta hacia afuera)
            if area > 1e-12:
                normal = normal / (2.0 * area)
            else:
                normal = np.array([0, 0, 1])
                # Advertir sobre triángulo degenerado
                print(f"   ⚠️ Panel {idx}: área casi cero")
            
            # Almacenar panel
            self.panels.append({
                'id': idx,
                'vertices': [v0, v1, v2],
                'center': center,
                'normal': normal,
                'area': area,
                'edge1': edge1,
                'edge2': edge2,
                'aspect_ratio': aspect_ratio,
                'skewness': skewness
            })
            
            # Almacenar métricas de calidad
            self.quality_metrics['areas'].append(area)
            self.quality_metrics['aspect_ratios'].append(aspect_ratio)
            self.quality_metrics['skewness'].append(skewness)
    
    def _validate_panels(self):
        """Validate panels and report quality metrics"""
        if len(self.panels) == 0:
            raise ValueError("No panels extracted from mesh")
        
        print("\n🔧 CALIDAD DE PANELES")
        print("-"*40)
        
        # Estadísticas de área
        areas = self.quality_metrics['areas']
        print(f"   Área promedio: {np.mean(areas):.8f} m²")
        print(f"   Área mínima: {np.min(areas):.8f} m²")
        print(f"   Área máxima: {np.max(areas):.8f} m²")
        
        # Paneles con área muy pequeña (degenerados)
        small_panels = np.sum(np.array(areas) < 1e-12)
        if small_panels > 0:
            print(f"   ⚠️ {small_panels} paneles con área casi cero")
        
        # Aspect Ratio
        ar = self.quality_metrics['aspect_ratios']
        print(f"   Aspect Ratio promedio: {np.mean(ar):.2f}")
        print(f"   Aspect Ratio máximo: {np.max(ar):.2f}")
        
        # Skewness
        sk = self.quality_metrics['skewness']
        print(f"   Skewness promedio: {np.mean(sk):.4f}")
        print(f"   Skewness máximo: {np.max(sk):.4f}")
        
        # Calidad general
        if np.max(ar) < 10 and np.max(sk) < 0.5:
            print("   ✅ Calidad de paneles: BUENA")
        elif np.max(ar) < 20 and np.max(sk) < 0.8:
            print("   ⚠️ Calidad de paneles: ACEPTABLE")
        else:
            print("   ❌ Calidad de paneles: MALA (recomienda refinar malla)")
        
        print("-"*40)
    
    # ============================================================
    # MÉTODOS DE ACCESO
    # ============================================================
    
    def get_panel_centers(self) -> np.ndarray:
        return np.array([p['center'] for p in self.panels])
    
    def get_panel_normals(self) -> np.ndarray:
        return np.array([p['normal'] for p in self.panels])
    
    def get_panel_areas(self) -> np.ndarray:
        return np.array([p['area'] for p in self.panels])
    
    def get_panel_vertices(self) -> List:
        return [p['vertices'] for p in self.panels]
    
    def get_center(self) -> np.ndarray:
        centers = self.get_panel_centers()
        return np.mean(centers, axis=0)
    
    def get_bounding_box(self) -> Tuple:
        centers = self.get_panel_centers()
        x_min, y_min, z_min = np.min(centers, axis=0)
        x_max, y_max, z_max = np.max(centers, axis=0)
        return (x_min, x_max, y_min, y_max, z_min, z_max)
    
    def get_quality_metrics(self) -> Dict:
        """Devuelve métricas de calidad de los paneles"""
        return self.quality_metrics
    
    def get_panels_summary(self) -> Dict:
        """Devuelve un resumen de los paneles"""
        return {
            'n_panels': len(self.panels),
            'area_mean': np.mean(self.quality_metrics['areas']),
            'area_min': np.min(self.quality_metrics['areas']),
            'area_max': np.max(self.quality_metrics['areas']),
            'aspect_ratio_mean': np.mean(self.quality_metrics['aspect_ratios']),
            'aspect_ratio_max': np.max(self.quality_metrics['aspect_ratios']),
            'skewness_mean': np.mean(self.quality_metrics['skewness']),
            'skewness_max': np.max(self.quality_metrics['skewness'])
        }
    
    @property
    def n_panels(self) -> int:
        return len(self.panels)