#!/usr/bin/env python3
"""
Matriz de Evaluación de Calidad de Servicio para Red CEB
Analiza fluctuaciones de potencia, tensión y calidad de servicio por nodo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ServiceQualityMatrix:
    def __init__(self, csv_path="../public/Mediciones Originales CEB .csv"):
        """Inicializar matriz de calidad de servicio"""
        self.df = pd.read_csv(csv_path)
        self.nodes_data = {}
        self.super_nodes = {}
        self.quality_matrix = {}
        
    def create_nodes_analysis(self):
        """Crear análisis detallado por nodo"""
        print("=== CREANDO ANÁLISIS POR NODO ===\n")
        
        # Procesar cada nodo (transformador/circuito)
        for _, row in self.df.dropna(subset=['Coordx', 'Coordy', 'POTENCIA', 'Usuarios Transformador']).iterrows():
            node_id = f"CT_{row['Codigoct']}"
            
            # Datos básicos del nodo
            self.nodes_data[node_id] = {
                'id': node_id,
                'subestacion': str(row.get('Historial Ago 21- May 25.Subestacion', 'UNKNOWN')),
                'alimentador': str(row.get('Idalimentador', 'UNKNOWN')),
                'coordx': row['Coordx'],
                'coordy': row['Coordy'],
                'potencia_instalada': row['POTENCIA'],
                'usuarios': row['Usuarios Transformador'],
                'direccion': row.get('Direccion', ''),
                
                # Indicadores de carga
                'densidad_usuarios': row['Usuarios Transformador'] / row['POTENCIA'] if row['POTENCIA'] > 0 else 0,
                'carga_especifica': row['POTENCIA'] / row['Usuarios Transformador'] if row['Usuarios Transformador'] > 0 else 0,
                
                # Estimaciones de fluctuaciones (basadas en patrones típicos)
                'fluctuacion_potencia_estimada': self._estimate_power_fluctuation(row),
                'fluctuacion_tension_estimada': self._estimate_voltage_fluctuation(row),
                'calidad_servicio_estimada': self._estimate_service_quality(row),
                
                # Factores de riesgo
                'factor_sobrecarga': self._calculate_overload_factor(row),
                'factor_estabilidad': self._calculate_stability_factor(row),
                'factor_confiabilidad': self._calculate_reliability_factor(row)
            }
        
        print(f"✅ Análisis completado para {len(self.nodes_data)} nodos")
        return self.nodes_data
    
    def _estimate_power_fluctuation(self, row):
        """Estimar fluctuaciones de potencia basado en características del nodo"""
        usuarios = row['Usuarios Transformador']
        potencia = row['POTENCIA']
        
        # Factores que afectan fluctuaciones de potencia
        density_factor = min(usuarios / potencia, 3.0) if potencia > 0 else 0  # Densidad de carga
        size_factor = np.log10(usuarios + 1) / 3.0  # Factor de tamaño
        geographic_factor = self._get_geographic_factor(row['Coordx'], row['Coordy'])
        
        # Fluctuación base (5-15% es típico en redes de distribución)
        base_fluctuation = 0.08  # 8% base
        
        # Fluctuación estimada considerando factores
        fluctuation = base_fluctuation * (1 + density_factor * 0.5 + size_factor * 0.3 + geographic_factor * 0.2)
        
        return min(fluctuation, 0.25)  # Máximo 25%
    
    def _estimate_voltage_fluctuation(self, row):
        """Estimar fluctuaciones de tensión basado en características del nodo"""
        usuarios = row['Usuarios Transformador']
        potencia = row['POTENCIA']
        
        # Las fluctuaciones de tensión están relacionadas con:
        # 1. Distancia a subestación (aproximada por coordenadas)
        # 2. Carga del transformador
        # 3. Tipo de carga (residencial/comercial/industrial)
        
        # Factor de distancia (asumiendo centro en coordenadas medias)
        center_x, center_y = -71.33, -41.13  # Centro aproximado de la red
        distance = np.sqrt((row['Coordx'] - center_x)**2 + (row['Coordy'] - center_y)**2)
        distance_factor = min(distance * 100, 2.0)  # Normalizar distancia
        
        # Factor de carga
        load_factor = min(usuarios / potencia, 2.0) if potencia > 0 else 0
        
        # Fluctuación base de tensión (2-8% es típico)
        base_voltage_fluctuation = 0.04  # 4% base
        
        # Fluctuación estimada
        voltage_fluctuation = base_voltage_fluctuation * (1 + distance_factor * 0.4 + load_factor * 0.3)
        
        return min(voltage_fluctuation, 0.12)  # Máximo 12%
    
    def _estimate_service_quality(self, row):
        """Estimar calidad de servicio (0-100, donde 100 es excelente)"""
        usuarios = row['Usuarios Transformador']
        potencia = row['POTENCIA']
        
        # Factores que afectan calidad de servicio
        ratio_usuarios_potencia = usuarios / potencia if potencia > 0 else 0
        
        # Calidad base
        base_quality = 85  # 85% base
        
        # Penalización por sobrecarga
        if ratio_usuarios_potencia > 1.5:
            overload_penalty = (ratio_usuarios_potencia - 1.5) * 20
        elif ratio_usuarios_potencia > 1.0:
            overload_penalty = (ratio_usuarios_potencia - 1.0) * 10
        else:
            overload_penalty = 0
        
        # Penalización por baja potencia instalada
        if potencia < 100 and usuarios > 50:
            low_power_penalty = 15
        else:
            low_power_penalty = 0
        
        # Bonificación por buena proporción
        if 0.3 <= ratio_usuarios_potencia <= 0.8:
            good_ratio_bonus = 10
        else:
            good_ratio_bonus = 0
        
        quality = base_quality - overload_penalty - low_power_penalty + good_ratio_bonus
        
        return max(min(quality, 100), 0)  # Entre 0 y 100
    
    def _calculate_overload_factor(self, row):
        """Calcular factor de sobrecarga (0-1, donde 1 es máxima sobrecarga)"""
        ratio = row['Usuarios Transformador'] / row['POTENCIA'] if row['POTENCIA'] > 0 else 0
        return min(ratio / 2.0, 1.0)  # Normalizar a 0-1
    
    def _calculate_stability_factor(self, row):
        """Calcular factor de estabilidad (0-1, donde 1 es más estable)"""
        # Transformadores más grandes tienden a ser más estables
        size_stability = min(row['POTENCIA'] / 500.0, 1.0)
        
        # Menor densidad de usuarios = mayor estabilidad
        density = row['Usuarios Transformador'] / row['POTENCIA'] if row['POTENCIA'] > 0 else 0
        density_stability = max(1.0 - density / 2.0, 0.0)
        
        return (size_stability + density_stability) / 2.0
    
    def _calculate_reliability_factor(self, row):
        """Calcular factor de confiabilidad basado en redundancia y ubicación"""
        # Simplificado: basado en tamaño y tipo de instalación
        if row['POTENCIA'] >= 500:
            return 0.9  # Alta confiabilidad
        elif row['POTENCIA'] >= 200:
            return 0.7  # Media confiabilidad
        else:
            return 0.5  # Baja confiabilidad
    
    def _get_geographic_factor(self, x, y):
        """Obtener factor geográfico que afecta fluctuaciones"""
        # Zonas más alejadas del centro tienden a tener más fluctuaciones
        center_x, center_y = -71.33, -41.13
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        return min(distance * 50, 1.0)  # Normalizar a 0-1
    
    def create_super_nodes(self, n_clusters=8):
        """Crear super nodos como agrupaciones geográficas"""
        print(f"\n=== CREANDO {n_clusters} SUPER NODOS GEOGRÁFICOS ===")
        
        if not self.nodes_data:
            self.create_nodes_analysis()
        
        # Preparar datos para clustering GEOGRÁFICO
        geographic_features = []
        node_ids = []
        
        for node_id, data in self.nodes_data.items():
            # Solo usar coordenadas geográficas para agrupación
            geographic_features.append([
                data['coordx'],
                data['coordy']
            ])
            node_ids.append(node_id)
        
        # Normalizar coordenadas geográficas
        scaler = StandardScaler()
        geographic_features_normalized = scaler.fit_transform(geographic_features)
        
        # Aplicar K-means basado en ubicación geográfica
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(geographic_features_normalized)
        
        # Crear super nodos
        for i in range(n_clusters):
            cluster_nodes = [node_ids[j] for j, cluster in enumerate(clusters) if cluster == i]
            
            if cluster_nodes:
                # Calcular estadísticas del super nodo
                cluster_data = [self.nodes_data[node_id] for node_id in cluster_nodes]
                
                self.super_nodes[f"ZONA_{i+1}"] = {
                    'id': f"ZONA_{i+1}",
                    'nombre_zona': self._get_zone_name(i+1, cluster_data),
                    'nodes_count': len(cluster_nodes),
                    'nodes': cluster_nodes,
                    'total_potencia': sum(d['potencia_instalada'] for d in cluster_data),
                    'total_usuarios': sum(d['usuarios'] for d in cluster_data),
                    'densidad_promedio': np.mean([d['densidad_usuarios'] for d in cluster_data]),
                    'area_geografica': self._calculate_geographic_area(cluster_data),
                    'densidad_espacial': len(cluster_nodes) / self._calculate_geographic_area(cluster_data),
                    'avg_fluctuacion_potencia': np.mean([d['fluctuacion_potencia_estimada'] for d in cluster_data]),
                    'avg_fluctuacion_tension': np.mean([d['fluctuacion_tension_estimada'] for d in cluster_data]),
                    'avg_calidad_servicio': np.mean([d['calidad_servicio_estimada'] for d in cluster_data]),
                    'criticidad_zona': self._calculate_zone_criticality(cluster_data),
                    'coordx_centro': np.mean([d['coordx'] for d in cluster_data]),
                    'coordy_centro': np.mean([d['coordy'] for d in cluster_data]),
                    'coordx_min': min(d['coordx'] for d in cluster_data),
                    'coordx_max': max(d['coordx'] for d in cluster_data),
                    'coordy_min': min(d['coordy'] for d in cluster_data),
                    'coordy_max': max(d['coordy'] for d in cluster_data),
                    'subestaciones_en_zona': len(set(d['subestacion'] for d in cluster_data))
                }
        
        print(f"✅ Creados {len(self.super_nodes)} super nodos")
        self._print_super_nodes_summary()
        
        return self.super_nodes
    
    def _get_zone_name(self, zone_num, cluster_data):
        """Generar nombre descriptivo para la zona geográfica"""
        center_x = np.mean([d['coordx'] for d in cluster_data])
        center_y = np.mean([d['coordy'] for d in cluster_data])
        
        # Determinar ubicación relativa
        if center_x > -71.25:
            ew_pos = "Este"
        elif center_x < -71.40:
            ew_pos = "Oeste"
        else:
            ew_pos = "Centro"
            
        if center_y > -41.10:
            ns_pos = "Norte"
        elif center_y < -41.20:
            ns_pos = "Sur"
        else:
            ns_pos = "Centro"
        
        return f"Zona {ew_pos}-{ns_pos}"
    
    def _calculate_geographic_area(self, cluster_data):
        """Calcular área geográfica del cluster en km²"""
        if len(cluster_data) < 2:
            return 0.1  # Área mínima
        
        coords_x = [d['coordx'] for d in cluster_data]
        coords_y = [d['coordy'] for d in cluster_data]
        
        # Aproximación simple: rectángulo que contiene todos los puntos
        width = (max(coords_x) - min(coords_x)) * 111  # 111 km por grado aprox
        height = (max(coords_y) - min(coords_y)) * 111
        
        area = width * height
        return max(area, 0.1)  # Área mínima 0.1 km²
    
    def _calculate_zone_criticality(self, cluster_data):
        """Calcular criticidad específica de la zona geográfica"""
        avg_quality = np.mean([d['calidad_servicio_estimada'] for d in cluster_data])
        avg_overload = np.mean([d['factor_sobrecarga'] for d in cluster_data])
        total_users = sum(d['usuarios'] for d in cluster_data)
        density = len(cluster_data) / self._calculate_geographic_area(cluster_data)
        
        # Criticidad considerando factores geográficos
        quality_factor = (100 - avg_quality) * 0.3
        overload_factor = avg_overload * 30
        users_factor = min(total_users / 1000, 20)
        density_factor = min(density * 10, 20)  # Penalizar alta densidad espacial
        
        criticality = quality_factor + overload_factor + users_factor + density_factor
        
        return min(criticality, 100)
    
    def _calculate_super_node_criticality(self, cluster_data):
        """Calcular criticidad del super nodo"""
        avg_quality = np.mean([d['calidad_servicio_estimada'] for d in cluster_data])
        avg_overload = np.mean([d['factor_sobrecarga'] for d in cluster_data])
        total_users = sum(d['usuarios'] for d in cluster_data)
        
        # Criticidad alta si: baja calidad, alta sobrecarga, muchos usuarios
        criticality = (100 - avg_quality) * 0.4 + avg_overload * 40 + min(total_users / 1000, 20)
        
        return min(criticality, 100)
    
    def _print_super_nodes_summary(self):
        """Mostrar resumen de zonas geográficas"""
        print("\n📊 RESUMEN DE ZONAS GEOGRÁFICAS:")
        sorted_zones = sorted(self.super_nodes.items(), 
                            key=lambda x: x[1]['criticidad_zona'], reverse=True)
        
        for zone_id, data in sorted_zones:
            print(f"\n{zone_id} ({data['nombre_zona']}) - Criticidad: {data['criticidad_zona']:.1f}")
            print(f"  📍 Nodos: {data['nodes_count']} | Área: {data['area_geografica']:.1f} km²")
            print(f"  👥 Usuarios: {data['total_usuarios']:,} | Densidad: {data['densidad_espacial']:.1f} nodos/km²")
            print(f"  ⚡ Potencia: {data['total_potencia']:.1f} kVA")
            print(f"  🏢 Subestaciones: {data['subestaciones_en_zona']}")
            print(f"  📈 Fluctuación Potencia: ±{data['avg_fluctuacion_potencia']*100:.1f}%")
            print(f"  📊 Fluctuación Tensión: ±{data['avg_fluctuacion_tension']*100:.1f}%")
            print(f"  🎯 Calidad Servicio: {data['avg_calidad_servicio']:.1f}%")
            print(f"  📍 Centro: ({data['coordx_centro']:.3f}, {data['coordy_centro']:.3f})")
    
    def analyze_power_voltage_correlation(self):
        """Analizar correlación entre fluctuaciones de potencia y tensión"""
        print("\n=== ANÁLISIS DE CORRELACIÓN POTENCIA-TENSIÓN ===")
        
        if not self.nodes_data:
            self.create_nodes_analysis()
        
        # Extraer datos para análisis
        power_fluctuations = [data['fluctuacion_potencia_estimada'] for data in self.nodes_data.values()]
        voltage_fluctuations = [data['fluctuacion_tension_estimada'] for data in self.nodes_data.values()]
        service_quality = [data['calidad_servicio_estimada'] for data in self.nodes_data.values()]
        
        # Calcular correlaciones
        corr_power_voltage = np.corrcoef(power_fluctuations, voltage_fluctuations)[0, 1]
        corr_power_quality = np.corrcoef(power_fluctuations, service_quality)[0, 1]
        corr_voltage_quality = np.corrcoef(voltage_fluctuations, service_quality)[0, 1]
        
        print(f"📊 Correlación Potencia-Tensión: {corr_power_voltage:.3f}")
        print(f"📊 Correlación Potencia-Calidad: {corr_power_quality:.3f}")
        print(f"📊 Correlación Tensión-Calidad: {corr_voltage_quality:.3f}")
        
        # Crear modelo de relación
        self._create_correlation_model(power_fluctuations, voltage_fluctuations, service_quality)
        
        return {
            'power_voltage_corr': corr_power_voltage,
            'power_quality_corr': corr_power_quality,
            'voltage_quality_corr': corr_voltage_quality
        }
    
    def _create_correlation_model(self, power_fluct, voltage_fluct, quality):
        """Crear modelo matemático de las relaciones"""
        print(f"\n🔬 MODELO DE RELACIONES:")
        
        # Modelo lineal simple: Calidad = f(Fluctuaciones)
        power_coef = np.polyfit(power_fluct, quality, 1)
        voltage_coef = np.polyfit(voltage_fluct, quality, 1)
        
        print(f"📐 Calidad = {power_coef[1]:.1f} + {power_coef[0]:.1f} × FluctPotencia")
        print(f"📐 Calidad = {voltage_coef[1]:.1f} + {voltage_coef[0]:.1f} × FluctTensión")
        
        # Modelo combinado
        X = np.column_stack((power_fluct, voltage_fluct))
        combined_coef = np.linalg.lstsq(X, quality, rcond=None)[0]
        
        print(f"📐 Calidad = Constante + {combined_coef[0]:.1f} × FluctPot + {combined_coef[1]:.1f} × FluctTens")
    
    def create_quality_matrix_visualization(self):
        """Crear visualizaciones de la matriz de calidad"""
        print("\n🎨 CREANDO VISUALIZACIONES...")
        
        # Configurar estilo
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Mapa de calidad de servicio
        ax1 = plt.subplot(2, 3, 1)
        self._plot_service_quality_map(ax1)
        
        # 2. Mapa de fluctuaciones de potencia
        ax2 = plt.subplot(2, 3, 2)
        self._plot_power_fluctuation_map(ax2)
        
        # 3. Mapa de fluctuaciones de tensión
        ax3 = plt.subplot(2, 3, 3)
        self._plot_voltage_fluctuation_map(ax3)
        
        # 4. Super nodos
        ax4 = plt.subplot(2, 3, 4)
        self._plot_super_nodes_map(ax4)
        
        # 5. Correlaciones
        ax5 = plt.subplot(2, 3, 5)
        self._plot_correlations(ax5)
        
        # 6. Matriz de criticidad
        ax6 = plt.subplot(2, 3, 6)
        self._plot_criticality_matrix(ax6)
        
        plt.tight_layout()
        plt.savefig('service_quality_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualización guardada: service_quality_matrix.png")
    
    def _plot_service_quality_map(self, ax):
        """Plotear mapa de calidad de servicio"""
        coords_x = [data['coordx'] for data in self.nodes_data.values()]
        coords_y = [data['coordy'] for data in self.nodes_data.values()]
        quality = [data['calidad_servicio_estimada'] for data in self.nodes_data.values()]
        
        scatter = ax.scatter(coords_x, coords_y, c=quality, cmap='RdYlGn', 
                           s=30, alpha=0.7, vmin=0, vmax=100)
        ax.set_title('Calidad de Servicio por Nodo', fontweight='bold')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        plt.colorbar(scatter, ax=ax, label='Calidad (%)')
        ax.grid(True, alpha=0.3)
    
    def _plot_power_fluctuation_map(self, ax):
        """Plotear mapa de fluctuaciones de potencia"""
        coords_x = [data['coordx'] for data in self.nodes_data.values()]
        coords_y = [data['coordy'] for data in self.nodes_data.values()]
        power_fluct = [data['fluctuacion_potencia_estimada'] * 100 for data in self.nodes_data.values()]
        
        scatter = ax.scatter(coords_x, coords_y, c=power_fluct, cmap='Reds', 
                           s=30, alpha=0.7)
        ax.set_title('Fluctuaciones de Potencia', fontweight='bold')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        plt.colorbar(scatter, ax=ax, label='Fluctuación (%)')
        ax.grid(True, alpha=0.3)
    
    def _plot_voltage_fluctuation_map(self, ax):
        """Plotear mapa de fluctuaciones de tensión"""
        coords_x = [data['coordx'] for data in self.nodes_data.values()]
        coords_y = [data['coordy'] for data in self.nodes_data.values()]
        voltage_fluct = [data['fluctuacion_tension_estimada'] * 100 for data in self.nodes_data.values()]
        
        scatter = ax.scatter(coords_x, coords_y, c=voltage_fluct, cmap='Blues', 
                           s=30, alpha=0.7)
        ax.set_title('Fluctuaciones de Tensión', fontweight='bold')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        plt.colorbar(scatter, ax=ax, label='Fluctuación (%)')
        ax.grid(True, alpha=0.3)
    
    def _plot_super_nodes_map(self, ax):
        """Plotear mapa de super nodos"""
        if not self.super_nodes:
            self.create_super_nodes()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.super_nodes)))
        
        for i, (sn_id, sn_data) in enumerate(self.super_nodes.items()):
            size = sn_data['criticidad_zona'] * 10  # Tamaño proporcional a criticidad
            ax.scatter(sn_data['coordx_centro'], sn_data['coordy_centro'], 
                      c=[colors[i]], s=size, alpha=0.8, edgecolors='black')
            ax.text(sn_data['coordx_centro'], sn_data['coordy_centro'], 
                   sn_id, fontsize=8, ha='center', va='center')
        
        ax.set_title('Super Nodos por Criticidad', fontweight='bold')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        ax.grid(True, alpha=0.3)
    
    def _plot_correlations(self, ax):
        """Plotear matriz de correlaciones"""
        power_fluct = [data['fluctuacion_potencia_estimada'] for data in self.nodes_data.values()]
        voltage_fluct = [data['fluctuacion_tension_estimada'] for data in self.nodes_data.values()]
        quality = [data['calidad_servicio_estimada'] for data in self.nodes_data.values()]
        
        ax.scatter(power_fluct, voltage_fluct, c=quality, cmap='RdYlGn', alpha=0.6)
        ax.set_xlabel('Fluctuación Potencia')
        ax.set_ylabel('Fluctuación Tensión')
        ax.set_title('Correlación Potencia-Tensión', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_criticality_matrix(self, ax):
        """Plotear matriz de criticidad"""
        if not self.super_nodes:
            return
        
        sn_names = list(self.super_nodes.keys())
        criticalities = [sn['criticidad_zona'] for sn in self.super_nodes.values()]
        
        bars = ax.bar(sn_names, criticalities, color='coral')
        ax.set_title('Criticidad por Super Nodo', fontweight='bold')
        ax.set_ylabel('Criticidad')
        ax.tick_params(axis='x', rotation=45)
        
        # Agregar valores en las barras
        for bar, crit in zip(bars, criticalities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{crit:.1f}', ha='center', va='bottom', fontsize=8)
    
    def generate_report(self):
        """Generar reporte completo de calidad de servicio"""
        print("\n=== GENERANDO REPORTE COMPLETO ===")
        
        # Ejecutar todos los análisis
        self.create_nodes_analysis()
        self.create_super_nodes()
        correlations = self.analyze_power_voltage_correlation()
        self.create_quality_matrix_visualization()
        
        # Identificar nodos más críticos
        critical_nodes = sorted(self.nodes_data.items(), 
                              key=lambda x: x[1]['calidad_servicio_estimada'])[:10]
        
        print(f"\n🚨 TOP 10 NODOS MÁS CRÍTICOS:")
        for node_id, data in critical_nodes:
            print(f"{node_id}: Calidad {data['calidad_servicio_estimada']:.1f}%, "
                  f"±{data['fluctuacion_potencia_estimada']*100:.1f}% pot, "
                  f"±{data['fluctuacion_tension_estimada']*100:.1f}% tens")
        
        # Guardar datos en JSON
        report_data = {
            'nodes_analysis': self.nodes_data,
            'super_nodes': self.super_nodes,
            'correlations': correlations,
            'critical_nodes': {node_id: data for node_id, data in critical_nodes}
        }
        
        with open('service_quality_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n✅ Reporte completo generado:")
        print(f"   - service_quality_matrix.png: Visualizaciones")
        print(f"   - service_quality_report.json: Datos detallados")

def main():
    """Función principal"""
    sqm = ServiceQualityMatrix()
    sqm.generate_report()

if __name__ == "__main__":
    main()