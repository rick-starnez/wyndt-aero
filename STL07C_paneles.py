import sys
import os
import numpy as np
import csv

# ============================================================
# IMPORT PANEL SOLVER
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.panel_solver import PanelSolver3D

# ============================================================
# IMPORTS
# ============================================================
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                             QSlider, QSpinBox, QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
import pyvista as pv
from pyvistaqt import QtInteractor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# FUNCIONES DE GRAFICACIÓN
# ============================================================

def plot_single_model(angles, Cl, Cd, torque, LD, 
                      model_name="Model", save_path=None):
    """Genera gráficas para un solo modelo"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Aerodynamic Results - {model_name}', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.plot(angles, Cl, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Angle θ (°)', fontsize=12)
    ax1.set_ylabel('Lift Coefficient (Cl)', fontsize=12)
    ax1.set_title('Cl vs Azimuth Angle', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 360)
    
    ax2 = axes[0, 1]
    ax2.plot(angles, Cd, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Angle θ (°)', fontsize=12)
    ax2.set_ylabel('Drag Coefficient (Cd)', fontsize=12)
    ax2.set_title('Cd vs Azimuth Angle', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 360)
    
    ax3 = axes[1, 0]
    ax3.plot(angles, LD, 'g-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Angle θ (°)', fontsize=12)
    ax3.set_ylabel('Efficiency (L/D)', fontsize=12)
    ax3.set_title('L/D vs Azimuth Angle', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 360)
    
    ax4 = axes[1, 1]
    ax4.plot(angles, torque, 'm-o', linewidth=2, markersize=6)
    ax4.set_xlabel('Angle θ (°)', fontsize=12)
    ax4.set_ylabel('Torque (N·m)', fontsize=12)
    ax4.set_title('Torque vs Azimuth Angle', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 360)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    plt.show()
    plt.close(fig)


# ============================================================
# CLASE PRINCIPAL
# ============================================================

class WindTunnelSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wind Tunnel - 3D Panel Solver")
        self.setGeometry(100, 100, 1550, 950)

        # ============================================================
        # VARIABLES
        # ============================================================
        self.stl_file = None
        self.theta = 0
        self.phi = 90        
        self.mesh_blade = None
        self.mesh_blade_original = None
        self.locator_vtk = None  
        
        self.velocity_mag = 16.0
        self.rho = 1.225
        self.scale_to_meters = 1.0 

        self.solver = None
        self.results = None
        self.sweep_in_progress = False
        self.sweep_angles = None
        self.sweep_torques = None
        self.V_inf_vec = None
        self.geometry_info = None
        
        self.last_sweep_angles = None
        self.last_sweep_Cl = None
        self.last_sweep_Cd = None
        self.last_sweep_torque = None
        self.last_sweep_LD = None
        self._current_fig = None
        self._diagnostic_fig = None
        
        # ============================================================
        # PERFORMANCE OPTIONS
        # ============================================================
        # Decimación DESACTIVADA (la malla ya es ligera)
        self.use_decimation = False
        self.decimation_factor = 0.5
        self.sweep_points = 18
        
        # Rango fijo para la escala de Cp
        self.cp_clim = [-5.0, 1.0]
        
        # ============================================================
        # SOLVER OPTIONS
        # ============================================================
        # NOTA: GMRES resultó ser 23× MÁS LENTO que el solver directo
        # para nuestras matrices densas. Por eso usamos siempre el directo.
        self.use_gmres = False

        # ============================================================
        # INTERFAZ GRÁFICA
        # ============================================================
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal: HORIZONTAL (panel izq + vista 3D der)
        main_layout = QHBoxLayout(central_widget)
        
        # ============================================================
        # PANEL IZQUIERDO: VERTICAL
        # ============================================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        # Labels internos (no visibles)
        self.lbl_info = QLabel("")
        self.lbl_dimensions = QLabel("")
        
        # ============================================================
        # BOTONES
        # ============================================================
        
        self.btn_load = QPushButton("📂 Load Wind Turbine STL")
        self.btn_load.setStyleSheet("""
            QPushButton { background-color: #27ae60; color: white; padding: 14px; font-weight: bold; border-radius: 4px; font-size: 13px; }
            QPushButton:hover { background-color: #2ecc71; }
        """)
        self.btn_load.clicked.connect(self.open_stl_direct)

        self.btn_plot_aero = QPushButton("📊 Plot Aerodynamic Curves")
        self.btn_plot_aero.setEnabled(False)
        self.btn_plot_aero.setStyleSheet("""
            QPushButton { background-color: #2980b9; color: white; padding: 14px; font-weight: bold; border-radius: 4px; font-size: 13px; }
            QPushButton:hover { background-color: #3498db; }
            QPushButton:disabled { background-color: #7f8c8d; }
        """)
        self.btn_plot_aero.clicked.connect(self.toggle_sweep)

        self.btn_diagnostics = QPushButton("🔬 Rotor Diagnostics")
        self.btn_diagnostics.setEnabled(False)
        self.btn_diagnostics.setStyleSheet("""
            QPushButton { background-color: #8e44ad; color: white; padding: 14px; font-weight: bold; border-radius: 4px; font-size: 13px; }
            QPushButton:hover { background-color: #9b59b6; }
            QPushButton:disabled { background-color: #7f8c8d; }
        """)
        self.btn_diagnostics.clicked.connect(self.show_diagnostics)

        # ============================================================
        # PROGRESS BAR
        # ============================================================
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 4px;
                background-color: #2c3e50;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 4px;
            }
        """)

        # ============================================================
        # POLAR ANGLE CONTROL
        # ============================================================
        spinbox_style = "background-color: #34495e; color: white; font-weight: bold; padding: 3px; border-radius: 3px;"
        
        self.lbl_theta = None
        self.slider_theta = None
        self.spin_theta = None

        # Label del Polar Angle
        self.lbl_phi = QLabel("Polar Angle (ϕ):")
        self.lbl_phi.setStyleSheet("color: #ecf0f1; font-weight: bold; font-size: 13px; padding-top: 10px;")
        
        # Slider horizontal
        self.slider_phi = QSlider(Qt.Orientation.Horizontal)
        self.slider_phi.setRange(0, 180)
        self.slider_phi.setValue(90)
        self.slider_phi.setEnabled(False)
        self.slider_phi.valueChanged.connect(self.sync_phi_from_slider)
        
        # Spinbox
        self.spin_phi = QSpinBox()
        self.spin_phi.setRange(0, 180)
        self.spin_phi.setValue(90)
        self.spin_phi.setEnabled(False)
        self.spin_phi.setSuffix(" °")
        self.spin_phi.setStyleSheet(spinbox_style)
        self.spin_phi.valueChanged.connect(self.sync_phi_from_spinbox)
        self.spin_phi.editingFinished.connect(self.on_phi_editing_finished)
        
        # ============================================================
        # Botón de actualizar vista
        # ============================================================
        self.btn_update_phi = QPushButton("🔄 Update View")
        self.btn_update_phi.setEnabled(False)
        self.btn_update_phi.setStyleSheet("""
            QPushButton { 
                background-color: #f39c12; 
                color: white; 
                padding: 10px; 
                font-weight: bold; 
                border-radius: 4px; 
                font-size: 13px; 
            }
            QPushButton:hover { background-color: #f1c40f; }
            QPushButton:disabled { background-color: #7f8c8d; }
        """)
        self.btn_update_phi.clicked.connect(self.on_phi_update_clicked)
        
        # Fila para el label y el spinbox
        phi_label_row = QHBoxLayout()
        phi_label_row.addWidget(self.lbl_phi)
        phi_label_row.addStretch()
        phi_label_row.addWidget(self.spin_phi)
        
        # Añadir todo al panel izquierdo
        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(self.btn_plot_aero)
        left_layout.addWidget(self.btn_diagnostics)
        left_layout.addWidget(self.progress_bar)
        left_layout.addLayout(phi_label_row)
        left_layout.addWidget(self.slider_phi)
        left_layout.addWidget(self.btn_update_phi)
        left_layout.addStretch()
        
        # Estilo del panel izquierdo
        left_panel.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
                border-radius: 6px;
            }
        """)
        left_panel.setFixedWidth(300)
        
        # ============================================================
        # VISTA 3D
        # ============================================================
        self.plotter = QtInteractor()
        self.plotter.background_color = "#111111"
        self.plotter.show_axes()
        self.plotter.view_isometric()
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.plotter.interactor, stretch=1)

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_value = 0
        self.cancel_requested = False

    # ============================================================
    # INTERFACE LOCKING
    # ============================================================
    
    def lock_interface(self):
        self.btn_load.setEnabled(False)
        self.btn_plot_aero.setEnabled(False)
        self.btn_diagnostics.setEnabled(False)
        self.btn_update_phi.setEnabled(False)
        if self.slider_phi is not None:
            self.slider_phi.setEnabled(False)
        if self.spin_phi is not None:
            self.spin_phi.setEnabled(False)
        self.progress_bar.setVisible(True)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def unlock_interface(self):
        self.btn_load.setEnabled(True)
        self.btn_plot_aero.setEnabled(True)
        self.btn_diagnostics.setEnabled(True)
        self.btn_update_phi.setEnabled(True)
        if self.slider_phi is not None:
            self.slider_phi.setEnabled(True)
        if self.spin_phi is not None:
            self.spin_phi.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.btn_plot_aero.setText("📊 Plot Aerodynamic Curves")
        self.btn_plot_aero.setStyleSheet("""
            QPushButton { background-color: #2980b9; color: white; padding: 14px; font-weight: bold; border-radius: 4px; font-size: 13px; }
            QPushButton:hover { background-color: #3498db; }
            QPushButton:disabled { background-color: #7f8c8d; }
        """)
        
        QApplication.restoreOverrideCursor()

    def toggle_sweep(self):
        if self.sweep_in_progress:
            self.cancel_requested = True
            print("⛔ Sweep cancellation requested...")
            QApplication.processEvents()
        else:
            self.plot_torque_curve()

    # ============================================================
    # CÁMARA SEGÚN LA DIRECCIÓN DEL VIENTO
    # ============================================================
    
    def _set_camera_from_phi(self):
        """Posiciona la cámara para mirar el rotor desde la dirección del viento."""
        try:
            rad_phi = np.radians(self.phi)
            
            wind_x = np.sin(rad_phi)
            wind_y = 0.0
            wind_z = np.cos(rad_phi)
            
            center = self.mesh_blade.center
            
            bounds = self.mesh_blade.bounds
            size = max(
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4]
            )
            distance = size * 3.0
            
            cam_x = center[0] - wind_x * distance
            cam_y = center[1] - wind_y * distance
            cam_z = center[2] - wind_z * distance
            
            lateral_offset = size * 0.3
            cam_x += lateral_offset * np.cos(rad_phi + np.pi/4)
            cam_y += lateral_offset * np.sin(rad_phi + np.pi/4)
            
            if abs(wind_z) > 0.95:
                up = (0, 1, 0)
            else:
                up = (0, 0, 1)
            
            self.plotter.camera_position = [
                (cam_x, cam_y, cam_z),
                center,
                up
            ]
            
            self.plotter.reset_camera()
            
            print(f"   📷 Cámara reposicionada para φ = {self.phi}°")
            
        except Exception as e:
            print(f"   ⚠️ Error al posicionar la cámara: {e}")
            self.plotter.reset_camera()
    
    # ============================================================
    # FORZAR RENDER EN MACOS
    # ============================================================
    
    def _force_render(self):
        """Fuerza el render de la vista 3D en macOS."""
        try:
            self.plotter.render_window.Render()
            
            widget = self.plotter.interactor
            current_size = widget.size()
            widget.resize(current_size.width() + 2, current_size.height() + 2)
            widget.resize(current_size.width(), current_size.height())
            
            QApplication.processEvents()
            
            print("   ✅ Vista actualizada")
        except Exception as e:
            print(f"   ⚠️ Error al renderizar: {e}")

    # ============================================================
    # GEOMETRY INSPECTION
    # ============================================================
    
    def inspect_geometry(self):
        if self.mesh_blade is None:
            return None
        
        print("\n" + "="*60)
        print("🔍 GEOMETRY INSPECTION")
        print("="*60)
        
        bounds = self.mesh_blade.bounds
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        
        print(f"Dimensions (original units): {dx:.4f} × {dy:.4f} × {dz:.4f}")
        
        center = np.array(self.mesh_blade.center)
        print(f"Center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        print(f"Number of triangles: {self.mesh_blade.n_cells}")
        
        mesh_prop = self.mesh_blade.compute_cell_sizes()
        area_total = np.sum(mesh_prop.cell_data['Area'])
        print(f"Total surface area: {area_total:.4f} units²")
        
        points = self.mesh_blade.points
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        quadrants = [0, 0, 0, 0]
        for p in points:
            if p[0] >= center_x and p[1] >= center_y:
                quadrants[0] += 1
            elif p[0] < center_x and p[1] >= center_y:
                quadrants[1] += 1
            elif p[0] < center_x and p[1] < center_y:
                quadrants[2] += 1
            else:
                quadrants[3] += 1
        
        print(f"\nPoint distribution by quadrant:")
        print(f"  Q1 (+,+): {quadrants[0]} ({quadrants[0]/len(points)*100:.1f}%)")
        print(f"  Q2 (-,+): {quadrants[1]} ({quadrants[1]/len(points)*100:.1f}%)")
        print(f"  Q3 (-,-): {quadrants[2]} ({quadrants[2]/len(points)*100:.1f}%)")
        print(f"  Q4 (+,-): {quadrants[3]} ({quadrants[3]/len(points)*100:.1f}%)")
        
        is_rotor = True
        if max(quadrants) / (min(quadrants) + 1) > 5:
            is_rotor = False
            print("\n⚠️ WARNING: Geometry does NOT appear to be a complete rotor")
        else:
            print("\n✅ Geometry appears to be a COMPLETE ROTOR")
        
        distances = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
        radius_max = np.max(distances)
        print(f"\nMaximum radius: {radius_max:.4f} units")
        
        print("="*60 + "\n")
        
        info = {
            'dx': dx, 'dy': dy, 'dz': dz,
            'center': center,
            'area_total': area_total,
            'radius_max': radius_max,
            'radius_m': radius_max * self.scale_to_meters,
            'is_rotor': is_rotor,
            'n_panels': self.mesh_blade.n_cells,
            'scale': self.scale_to_meters
        }
        
        self.geometry_info = info
        return info

    # ============================================================
    # MAIN FUNCTIONS
    # ============================================================
    
    def open_stl_direct(self):
        file, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Wind Turbine STL", 
            "", 
            "STL Files (*.stl);;All Files (*)"
        )
        if file: 
            self.load_geometry(file)

    def load_geometry(self, file_path):
        self.stl_file = file_path
        try:
            self.plotter.clear()
            raw_mesh = pv.read(file_path)
            
            self.mesh_blade_original = raw_mesh
            self.mesh_blade = raw_mesh.triangulate() if not raw_mesh.is_all_triangles else raw_mesh
            
            raw_bounds = self.mesh_blade.bounds
            dx_raw = raw_bounds[1] - raw_bounds[0]
            dy_raw = raw_bounds[3] - raw_bounds[2]
            dz_raw = raw_bounds[5] - raw_bounds[4]
            max_dim_raw = max(dx_raw, dy_raw, dz_raw)
            
            if max_dim_raw > 10:
                self.scale_to_meters = 0.001
                unit_str = "mm"
                print(f"🔧 DETECTED: {max_dim_raw:.1f} mm → {max_dim_raw * 0.001:.3f} m")
            elif max_dim_raw > 5.0:
                self.scale_to_meters = 0.001
                unit_str = "mm (detected)"
            else:
                self.scale_to_meters = 1.0
                unit_str = "m (detected)"
            
            if self.scale_to_meters != 1.0:
                scaled_points = self.mesh_blade.points * self.scale_to_meters
                self.mesh_blade.points = scaled_points
                print(f"🔧 Points scaled to meters (factor {self.scale_to_meters})")
            
            if self.use_decimation and self.mesh_blade.n_cells > 10000:
                original_panels = self.mesh_blade.n_cells
                self.mesh_blade = self.mesh_blade.decimate(self.decimation_factor)
                new_panels = self.mesh_blade.n_cells
                print(f"🔧 Decimation: {original_panels} → {new_panels} panels ({(new_panels/original_panels*100):.0f}%)")

            self.locator_vtk = pv._vtk.vtkCellLocator()
            self.locator_vtk.SetDataSet(self.mesh_blade)
            self.locator_vtk.BuildLocator()

            bounds = self.mesh_blade.bounds
            dx_m = bounds[1] - bounds[0]
            dy_m = bounds[3] - bounds[2]
            dz_m = bounds[5] - bounds[4]
            
            center = np.array(self.mesh_blade.center)
            points = self.mesh_blade.points
            distances = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
            radius_m = np.max(distances)
            
            print(f"\n📏 DIMENSIONS (SI):")
            print(f"   {dx_m:.3f}m × {dy_m:.3f}m × {dz_m:.3f}m")
            print(f"   Rotor: R={radius_m:.3f}m")
            print(f"   Original: {max_dim_raw:.1f} {unit_str}")

            self.plotter.add_mesh(
                self.mesh_blade, 
                color="#ffffff", 
                smooth_shading=True, 
                opacity=0.25, 
                show_edges=False
            )
            
            self._set_camera_from_phi()
            self._force_render()
            
            info = self.inspect_geometry()
            
            if info and not info['is_rotor']:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle("⚠️ Geometry Warning")
                msg.setText(
                    "The loaded geometry does NOT appear to be a complete rotor.\n\n"
                    f"Dimensions: {info['dx']:.3f} × {info['dy']:.3f} × {info['dz']:.3f} m\n"
                    f"Maximum radius: {info['radius_max']:.3f} m\n"
                    "If this is an INDIVIDUAL BLADE, torque calculation is NOT representative."
                )
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.exec()
            
            if self.slider_phi is not None:
                self.slider_phi.setEnabled(True)
            if self.spin_phi is not None:
                self.spin_phi.setEnabled(True)
            self.btn_update_phi.setEnabled(True)
            self.btn_plot_aero.setEnabled(True)
            self.btn_diagnostics.setEnabled(True)
            
            self.compute_flow()
            
        except Exception as e:
            print(f"❌ Error loading geometry: {str(e)}")
            import traceback
            traceback.print_exc()

    # ============================================================
    # PHI CONTROL HANDLERS
    # ============================================================
    
    def sync_phi_from_slider(self, value):
        self.phi = value
        if self.spin_phi is not None:
            self.spin_phi.blockSignals(True)
            self.spin_phi.setValue(value)
            self.spin_phi.blockSignals(False)

    def sync_phi_from_spinbox(self, value):
        self.phi = value
        if self.slider_phi is not None:
            self.slider_phi.blockSignals(True)
            self.slider_phi.setValue(value)
            self.slider_phi.blockSignals(False)

    def on_phi_update_clicked(self):
        if self.mesh_blade is None or self.sweep_in_progress:
            return
        
        print(f"\n🔄 Updating view for φ = {self.phi}°...")
        
        self.btn_update_phi.setText("⏳ Updating...")
        self.btn_update_phi.setEnabled(False)
        QApplication.processEvents()
        
        try:
            self.compute_flow()
        finally:
            self.btn_update_phi.setText("🔄 Update View")
            self.btn_update_phi.setEnabled(True)
            QApplication.processEvents()

    def on_phi_editing_finished(self):
        if self.mesh_blade is None or self.sweep_in_progress:
            return
        
        print(f"\n🔄 Updating view for φ = {self.phi}° (from spinbox)...")
        
        self.btn_update_phi.setText("⏳ Updating...")
        self.btn_update_phi.setEnabled(False)
        QApplication.processEvents()
        
        try:
            self.compute_flow()
        finally:
            self.btn_update_phi.setText("🔄 Update View")
            self.btn_update_phi.setEnabled(True)
            QApplication.processEvents()

    # ============================================================
    # COMPUTE FLOW
    # ============================================================
    
    def compute_flow(self):
        if self.mesh_blade is None:
            return
        self._compute_with_panels()

    def _compute_with_panels(self):
        print("🔬 Solving potential flow...")
        QApplication.processEvents()
        
        try:
            self.solver = PanelSolver3D(
                self.mesh_blade, 
                escala=1.0,
                rho=self.rho
            )
            
            # ============================================================
            # SOLVER: SIEMPRE USAR DIRECTO (GMRES es 23× más lento)
            # ============================================================
            self.results = self.solver.solve(
                V_inf_magnitude=self.velocity_mag,
                theta=self.theta,
                phi=self.phi,
                use_gmres=False
            )
            
            rad_theta = np.radians(self.theta)
            rad_phi = np.radians(self.phi)
            self.V_inf_vec = np.array([
                self.velocity_mag * np.sin(rad_phi) * np.cos(rad_theta),
                self.velocity_mag * np.sin(rad_phi) * np.sin(rad_theta),
                self.velocity_mag * np.cos(rad_phi)
            ])
            
            self.plotter.clear()
            
            self.mesh_blade.cell_data['Cp'] = self.results['Cp']
            self.plotter.add_mesh(
                self.mesh_blade, 
                scalars='Cp',
                cmap='coolwarm',
                opacity=0.85,
                show_edges=False,
                lighting=True,
                clim=self.cp_clim,
                scalar_bar_args={
                    'title': 'Pressure Coefficient (Cp)',
                    'title_font_size': 11,
                    'label_font_size': 9
                }
            )
            
            self._set_camera_from_phi()
            self._force_render()
            
            Cl_raw = self.results['forces']['Cl']
            Cd_raw = self.results['forces']['Cd']
            torque_raw = self.results['forces']['torque_z']
            
            effective_angle = self.theta % 360
            if effective_angle > 180:
                angle_of_attack = 360 - effective_angle
            else:
                angle_of_attack = effective_angle
            
            if angle_of_attack < 5:
                correction = 0.3 + 0.04 * angle_of_attack
            elif angle_of_attack < 20:
                correction = 0.5 + 0.025 * (angle_of_attack - 5)
                correction = min(correction, 0.9)
            elif angle_of_attack < 30:
                correction = 0.9 - 0.02 * (angle_of_attack - 20)
            else:
                correction = 0.7 * np.exp(-0.05 * (angle_of_attack - 30))
                correction = max(correction, 0.05)
            
            Cl_corr = abs(Cl_raw) * correction
            Cd_corr = abs(Cd_raw) * correction * 0.4
            Cl_corr = max(0.005, Cl_corr)
            Cd_corr = max(0.002, Cd_corr)
            torque_corr = abs(torque_raw) * 0.015
            
            print(f"\n🔧 CORRECTED RESULTS:")
            print(f"   Effective angle of attack: {angle_of_attack:.1f}°")
            print(f"   Correction factor: {correction:.3f}")
            print(f"   Cl raw: {Cl_raw:.4f} → corrected: {Cl_corr:.4f}")
            print(f"   Cd raw: {Cd_raw:.4f} → corrected: {Cd_corr:.4f}")
            print(f"   Torque raw: {torque_raw:.4f} → corrected: {torque_corr:.4f} N·m")
            
            self._force_render()
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

    # ============================================================
    # TORQUE CURVE (barrido 360° CON CACHÉ REUTILIZADO)
    # ============================================================
    
    def plot_torque_curve(self):
        if self.mesh_blade is None:
            return
        
        if self.sweep_in_progress:
            return
        
        if self.geometry_info and not self.geometry_info['is_rotor']:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("⚠️ Warning")
            msg.setText(
                "This geometry does NOT appear to be a complete rotor.\n\n"
                "Torque calculation for an individual blade is NOT representative.\n"
                "Do you want to continue anyway?"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            reply = msg.exec()
            if reply == QMessageBox.StandardButton.No:
                return
        
        self.sweep_in_progress = True
        self.cancel_requested = False
        
        self.btn_plot_aero.setText("⛔ Cancel Sweep")
        self.btn_plot_aero.setStyleSheet("""
            QPushButton { background-color: #e74c3c; color: white; padding: 14px; font-weight: bold; border-radius: 4px; font-size: 13px; }
            QPushButton:hover { background-color: #c0392b; }
            QPushButton:disabled { background-color: #7f8c8d; }
        """)
        
        self.lock_interface()
        print("🔄 Computing aerodynamic curves...")
        QApplication.processEvents()
        
        try:
            angles = np.linspace(0, 360, self.sweep_points)
            torque_values = []
            Cl_values = []
            Cd_values = []
            
            total = len(angles)
            
            print("\n" + "="*60)
            print("📊 AERODYNAMIC SWEEP RESULTS")
            print("="*60)
            print(f"{'Angle θ (°)':<12} {'Cl':<10} {'Cd':<10} {'Torque (N·m)':<15} {'L/D':<10}")
            print("-"*60)
            
            # ============================================================
            # Crear UN SOLO solver para todo el barrido
            # ============================================================
            print(f"🔧 Inicializando solver único para el barrido ({self.mesh_blade.n_cells} paneles)...")
            QApplication.processEvents()
            
            sweep_solver = PanelSolver3D(
                self.mesh_blade,
                escala=1.0,
                rho=self.rho,
                verbose=True
            )
            
            # Forzar la construcción de la matriz una sola vez
            _ = sweep_solver._get_influence_data()
            print(f"✅ Matriz de influencia construida. Ahora el barrido será rápido.\n")
            QApplication.processEvents()
            
            for i, theta in enumerate(angles):
                if self.cancel_requested:
                    print("⛔ Sweep cancelled by user")
                    break
                
                progress = int((i + 1) / total * 100)
                self.progress_bar.setValue(progress)
                print(f"🔄 {i+1}/{total} (θ={theta:.1f}°) | Paneles: {sweep_solver.n_panels}")
                QApplication.processEvents()
                
                # ============================================================
                # SOLVER: SIEMPRE USAR DIRECTO
                # ============================================================
                results = sweep_solver.solve(
                    V_inf_magnitude=self.velocity_mag,
                    theta=theta,
                    phi=self.phi,
                    use_gmres=False
                )
                
                Cl_raw = results['forces']['Cl']
                Cd_raw = results['forces']['Cd']
                torque_raw = results['forces']['torque_z']
                
                effective_angle = theta % 360
                if effective_angle > 180:
                    angle_of_attack = 360 - effective_angle
                else:
                    angle_of_attack = effective_angle
                
                if angle_of_attack < 5:
                    correction = 0.3 + 0.04 * angle_of_attack
                elif angle_of_attack < 20:
                    correction = 0.5 + 0.025 * (angle_of_attack - 5)
                    correction = min(correction, 0.9)
                elif angle_of_attack < 30:
                    correction = 0.9 - 0.02 * (angle_of_attack - 20)
                else:
                    correction = 0.7 * np.exp(-0.05 * (angle_of_attack - 30))
                    correction = max(correction, 0.05)
                
                Cl_corr = abs(Cl_raw) * correction
                Cd_corr = abs(Cd_raw) * correction * 0.4
                Cl_corr = max(0.005, Cl_corr)
                Cd_corr = max(0.002, Cd_corr)
                torque_corr = abs(torque_raw) * 0.015
                
                torque_values.append(torque_corr)
                Cl_values.append(Cl_corr)
                Cd_values.append(Cd_corr)
                
                ld = Cl_corr / Cd_corr if Cd_corr > 0 else 0
                print(f"{theta:<12.1f} {Cl_corr:<10.4f} {Cd_corr:<10.4f} {torque_corr:<15.4f} {ld:<10.2f}")
            
            print("="*60 + "\n")
            
            if self.cancel_requested:
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(False)
                self.sweep_in_progress = False
                self.unlock_interface()
                return
            
            self.last_sweep_angles = angles[:len(torque_values)]
            self.last_sweep_Cl = np.array(Cl_values)
            self.last_sweep_Cd = np.array(Cd_values)
            self.last_sweep_torque = np.array(torque_values)
            self.last_sweep_LD = np.array([
                Cl_values[i]/Cd_values[i] if Cd_values[i] > 0 else 0 
                for i in range(len(torque_values))
            ])
            
            self._show_results_figure()
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.sweep_in_progress = False
            self.unlock_interface()

    def _show_results_figure(self):
        """Muestra la figura con las 4 gráficas aerodinámicas"""
        model_name = "Model"
        if self.stl_file:
            model_name = os.path.splitext(os.path.basename(self.stl_file))[0]
        
        if self._current_fig is not None:
            plt.close(self._current_fig)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Aerodynamic Results - {model_name}', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        ax1.plot(self.last_sweep_angles, self.last_sweep_Cl, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Angle θ (°)', fontsize=12)
        ax1.set_ylabel('Lift Coefficient (Cl)', fontsize=12)
        ax1.set_title('Cl vs Azimuth Angle', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 360)
        
        ax2 = axes[0, 1]
        ax2.plot(self.last_sweep_angles, self.last_sweep_Cd, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Angle θ (°)', fontsize=12)
        ax2.set_ylabel('Drag Coefficient (Cd)', fontsize=12)
        ax2.set_title('Cd vs Azimuth Angle', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 360)
        
        ax3 = axes[1, 0]
        ax3.plot(self.last_sweep_angles, self.last_sweep_LD, 'g-o', linewidth=2, markersize=6)
        ax3.set_xlabel('Angle θ (°)', fontsize=12)
        ax3.set_ylabel('Efficiency (L/D)', fontsize=12)
        ax3.set_title('L/D vs Azimuth Angle', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 360)
        
        ax4 = axes[1, 1]
        ax4.plot(self.last_sweep_angles, self.last_sweep_torque, 'm-o', linewidth=2, markersize=6)
        ax4.set_xlabel('Angle θ (°)', fontsize=12)
        ax4.set_ylabel('Torque (N·m)', fontsize=12)
        ax4.set_title('Torque vs Azimuth Angle', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 360)
        
        plt.tight_layout()
        plt.show(block=False)
        
        self._current_fig = fig
        print(f"📊 Figure displayed for {model_name}")

    # ============================================================
    # ROTOR DIAGNOSTICS
    # ============================================================
    
    def show_diagnostics(self):
        """Genera las 6 gráficas de diagnóstico del rotor"""
        if self.mesh_blade is None or self.results is None:
            QMessageBox.warning(self, "No Data", 
                "Please load a geometry and compute the flow first.")
            return
        
        print("\n" + "="*60)
        print("🔬 ROTOR DIAGNOSTICS")
        print("="*60)
        
        try:
            centers = self.solver.geometry.get_panel_centers()
            normals = self.solver.geometry.get_panel_normals()
            areas = self.solver.geometry.get_panel_areas()
            
            Cp = self.results['Cp']
            
            p_gauge = self.solver.pressures - 101325.0
            panel_forces = -p_gauge[:, np.newaxis] * areas[:, np.newaxis] * normals
            
            torque_panel = np.zeros(len(centers))
            for i in range(len(centers)):
                r_vec = np.array([centers[i, 0], centers[i, 1], 0])
                torque_vec = np.cross(r_vec, panel_forces[i])
                torque_panel[i] = torque_vec[2]
            
            angulos = np.degrees(np.arctan2(centers[:, 1], centers[:, 0])) % 360
            
            if self._diagnostic_fig is not None:
                plt.close(self._diagnostic_fig)
            
            fig = plt.figure(figsize=(16, 10))
            
            # --- 1. Vista 3D del Rotor ---
            ax1 = fig.add_subplot(2, 3, 1, projection='3d')
            ax1.scatter(centers[::10, 0], centers[::10, 1], centers[::10, 2], 
                        c='blue', s=1, alpha=0.5)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D View of Rotor', fontsize=12, fontweight='bold')
            
            # --- 2. Distribución angular de celdas ---
            ax2 = fig.add_subplot(2, 3, 2)
            ax2.hist(angulos, bins=72, color='blue', alpha=0.7)
            ax2.set_xlabel('Azimuthal Angle (°)')
            ax2.set_ylabel('Number of Cells')
            ax2.set_title('Angular Distribution of Cells', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # --- 3. Distribución en Z (espesor) ---
            ax3 = fig.add_subplot(2, 3, 3)
            ax3.hist(centers[:, 2], bins=50, color='green', alpha=0.7)
            ax3.set_xlabel('Z (m)')
            ax3.set_ylabel('Number of Cells')
            ax3.set_title('Distribution in Z (Thickness)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # --- 4. Distribución de Cp ---
            ax4 = fig.add_subplot(2, 3, 4)
            scatter4 = ax4.scatter(centers[:, 0], centers[:, 1],
                                   c=Cp, cmap='coolwarm',
                                   s=2, alpha=0.7,
                                   vmin=np.percentile(Cp, 1),
                                   vmax=np.percentile(Cp, 99))
            plt.colorbar(scatter4, ax=ax4, label='Cp')
            ax4.set_xlabel('X (m)')
            ax4.set_ylabel('Y (m)')
            ax4.set_title(f'Cp Distribution (θ={self.theta}°)', fontsize=12, fontweight='bold')
            ax4.set_aspect('equal')
            ax4.grid(True, alpha=0.3)
            
            # --- 5. Torque por panel ---
            ax5 = fig.add_subplot(2, 3, 5)
            max_torque = np.max(np.abs(torque_panel)) if len(torque_panel) > 0 else 1.0
            scatter5 = ax5.scatter(centers[:, 0], centers[:, 1],
                                   c=torque_panel, cmap='RdBu_r',
                                   s=2, alpha=0.7,
                                   vmin=-max_torque, vmax=max_torque)
            plt.colorbar(scatter5, ax=ax5, label='Torque per Panel (N·m)')
            ax5.set_xlabel('X (m)')
            ax5.set_ylabel('Y (m)')
            ax5.set_title('Torque per Panel', fontsize=12, fontweight='bold')
            ax5.set_aspect('equal')
            ax5.grid(True, alpha=0.3)
            
            # --- 6. Torque por sector angular ---
            ax6 = fig.add_subplot(2, 3, 6)
            sectores = 12
            torque_sectores = []
            angulos_medios = []
            for s in range(sectores):
                ang_ini = s * (360 / sectores)
                ang_fin = (s + 1) * (360 / sectores)
                mask = (angulos >= ang_ini) & (angulos < ang_fin)
                torque_sectores.append(np.sum(torque_panel[mask]))
                angulos_medios.append((ang_ini + ang_fin) / 2)
            
            colors = ['red' if t > 0 else 'blue' for t in torque_sectores]
            ax6.bar(angulos_medios, torque_sectores, width=25, color=colors, alpha=0.7)
            ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax6.set_xlabel('Azimuthal Angle (°)')
            ax6.set_ylabel('Torque per Sector (N·m)')
            ax6.set_title('Torque per Angular Sector', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show(block=False)
            
            self._diagnostic_fig = fig
            
            torque_total = np.sum(torque_panel)
            torque_pos = np.sum(torque_panel[torque_panel > 0])
            torque_neg = np.sum(torque_panel[torque_panel < 0])
            
            print(f"\n📊 DIAGNOSTIC SUMMARY:")
            print(f"   Panels: {len(centers)}")
            print(f"   Cp range: [{np.min(Cp):.4f}, {np.max(Cp):.4f}]")
            print(f"   Torque total: {torque_total:.6f} N·m")
            print(f"   Torque positive: +{torque_pos:.6f} N·m")
            print(f"   Torque negative: {torque_neg:.6f} N·m")
            print(f"   Cancellation: {abs(torque_pos + torque_neg) / (abs(torque_pos) + abs(torque_neg) + 1e-10) * 100:.1f}%")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Error generating diagnostics: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Could not generate diagnostics:\n{str(e)}")

    def update_progress(self):
        self.progress_bar.setValue(self.progress_value)

    # ============================================================
    # EXTRACT CP FROM SPHERE
    # ============================================================
    
    def extract_sphere_cp(self):
        if self.mesh_blade is None:
            QMessageBox.warning(self, "No Geometry", "Please load a sphere STL first.")
            return
        
        info = self.geometry_info
        if info:
            radius = info['radius_max']
            if abs(radius - 1.0) < 0.1:
                print("\n" + "="*50)
                print("🔵 SPHERE CP EXTRACTION")
                print("="*50)
                print(f"   Radius: {radius:.4f} m")
                print(f"   Number of panels: {info['n_panels']}")
                
                Cp = self.results['Cp']
                print(f"   Cp max: {np.max(Cp):.4f}")
                print(f"   Cp min: {np.min(Cp):.4f}")
                print(f"   Cp mean: {np.mean(Cp):.4f}")
                print("="*50)
                
                return Cp
            else:
                QMessageBox.warning(self, "Not a Sphere", 
                    f"The loaded geometry has radius {radius:.3f}m, not 1.0m.\n"
                    "Please load a sphere with radius 1.0 for validation.")
                return None
        else:
            QMessageBox.warning(self, "No Geometry Info", "Please load a sphere STL first.")
            return None

    # ============================================================
    # CLOSE EVENT
    # ============================================================
    def closeEvent(self, event):
        if hasattr(self, '_current_fig') and self._current_fig is not None:
            plt.close(self._current_fig)
        if hasattr(self, '_diagnostic_fig') and self._diagnostic_fig is not None:
            plt.close(self._diagnostic_fig)
        self.plotter.close()
        event.accept()


# ============================================================
# MAIN
# ============================================================
def main():
    app = QApplication(sys.argv)
    window = WindTunnelSimulator()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()