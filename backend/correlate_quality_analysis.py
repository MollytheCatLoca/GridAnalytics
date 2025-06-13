#!/usr/bin/env python3
"""
Análisis de Correlación entre Calidad de Servicio Estimada vs Datos Originales CEB
Compara nuestra matriz de calidad con los datos reales de mediciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import json
from service_quality_matrix import ServiceQualityMatrix

class QualityCorrelationAnalysis:
    def __init__(self, csv_path="../public/Mediciones Originales CEB .csv"):
        """Inicializar análisis de correlación"""
        self.df = pd.read_csv(csv_path)
        self.sqm = ServiceQualityMatrix(csv_path)
        self.correlation_results = {}
        
    def load_estimated_quality_data(self):
        """Cargar datos de calidad estimada generados por nuestro modelo"""
        print("=== CARGANDO DATOS DE CALIDAD ESTIMADA ===\n")
        
        # Ejecutar análisis de calidad de servicio
        self.sqm.create_nodes_analysis()
        
        print(f"✅ Datos de calidad estimada cargados: {len(self.sqm.nodes_data)} nodos")
        return self.sqm.nodes_data
    
    def analyze_original_quality_indicators(self):
        """Analizar indicadores de calidad en datos originales"""
        print("=== ANALIZANDO INDICADORES ORIGINALES DE CALIDAD ===\n")
        
        # Columnas relevantes para calidad de servicio
        quality_columns = [
            'Resultado',           # Resultado de medición (exitosa, fallida, penalizada)
            'Tipo de medicion',    # Tipo de medición
            'Periodo',            # Período de medición
            'Fecha_Colocacion',   # Fecha de colocación del medidor
            'Fecha_Retiro',       # Fecha de retiro del medidor
            'Nro_EPRE'           # Número EPRE (posible indicador regulatorio)
        ]
        
        print("📊 ANÁLISIS DE COLUMNAS DE CALIDAD:")
        original_quality_data = {}
        
        for col in quality_columns:
            if col in self.df.columns:
                print(f"\n{col}:")
                unique_vals = self.df[col].value_counts()
                print(unique_vals.head(10))
                
                # Almacenar datos para correlación
                original_quality_data[col] = self.df[col].fillna('Sin datos')
        
        # Análisis específico de 'Resultado' como indicador principal
        if 'Resultado' in self.df.columns:
            print(f"\n🎯 ANÁLISIS DETALLADO DE 'RESULTADO':")
            result_analysis = self.df.groupby('Resultado').agg({
                'POTENCIA': ['count', 'mean', 'sum'],
                'Usuarios Transformador': ['mean', 'sum'],
                'Coordx': 'count'
            }).round(2)
            print(result_analysis)
        
        return original_quality_data
    
    def create_quality_scoring_from_original(self):
        """Crear puntuación de calidad basada en datos originales"""
        print("\n=== CREANDO PUNTUACIÓN DE CALIDAD ORIGINAL ===\n")
        
        quality_scores = {}
        
        # Procesar cada fila para crear score de calidad
        for _, row in self.df.iterrows():
            node_id = f"CT_{row['Codigoct']}"
            
            # Score base
            base_score = 50
            
            # Factor por resultado de medición
            resultado = str(row.get('Resultado', 'Sin datos'))
            if resultado == 'Exitosa':
                result_bonus = 30
            elif resultado == 'Fallida':
                result_bonus = -20
            elif resultado == 'Penalizada':
                result_bonus = -10
            elif resultado == 'Sin medicion':
                result_bonus = -15
            else:
                result_bonus = 0
            
            # Factor por presencia de medidor EPRE
            epre = str(row.get('Nro_EPRE', 'Sin Medir'))
            if epre != 'Sin Medir' and epre != 'nan':
                epre_bonus = 15
            else:
                epre_bonus = -5
            
            # Factor por antigüedad (si hay fecha de inicio)
            fecha_inicio = str(row.get('Fechainicio', ''))
            if fecha_inicio and fecha_inicio != 'nan':
                try:
                    year = int(fecha_inicio.split('-')[0])
                    if year >= 2020:
                        age_bonus = 10
                    elif year >= 2015:
                        age_bonus = 5
                    elif year >= 2010:
                        age_bonus = 0
                    else:
                        age_bonus = -5
                except:
                    age_bonus = 0
            else:
                age_bonus = -5
            
            # Score final
            final_score = base_score + result_bonus + epre_bonus + age_bonus
            final_score = max(0, min(100, final_score))  # Entre 0 y 100
            
            quality_scores[node_id] = {
                'calidad_original': final_score,
                'resultado_medicion': resultado,
                'tiene_epre': epre != 'Sin Medir',
                'antiguedad_estimada': fecha_inicio[:4] if fecha_inicio and fecha_inicio != 'nan' else 'Sin datos'
            }
        
        print(f"✅ Puntuaciones de calidad original calculadas: {len(quality_scores)} nodos")
        
        # Estadísticas de calidad original
        scores = [data['calidad_original'] for data in quality_scores.values()]
        print(f"📊 Estadísticas calidad original:")
        print(f"   - Promedio: {np.mean(scores):.1f}")
        print(f"   - Mediana: {np.median(scores):.1f}")
        print(f"   - Desviación: {np.std(scores):.1f}")
        print(f"   - Rango: {min(scores):.1f} - {max(scores):.1f}")
        
        return quality_scores
    
    def correlate_estimated_vs_original(self, estimated_data, original_scores):
        """Correlacionar calidad estimada vs original"""
        print("\n=== CORRELACIÓN CALIDAD ESTIMADA VS ORIGINAL ===\n")
        
        # Encontrar nodos comunes
        common_nodes = set(estimated_data.keys()) & set(original_scores.keys())
        print(f"📊 Nodos comunes para análisis: {len(common_nodes)}")
        
        if len(common_nodes) < 10:
            print("⚠️ Pocos nodos comunes para análisis estadístico robusto")
            return None
        
        # Extraer datos para correlación
        estimated_quality = []
        original_quality = []
        node_ids = []
        
        for node_id in common_nodes:
            estimated_quality.append(estimated_data[node_id]['calidad_servicio_estimada'])
            original_quality.append(original_scores[node_id]['calidad_original'])
            node_ids.append(node_id)
        
        # Calcular correlaciones
        pearson_corr, pearson_p = pearsonr(estimated_quality, original_quality)
        spearman_corr, spearman_p = spearmanr(estimated_quality, original_quality)
        
        print(f"📊 RESULTADOS DE CORRELACIÓN:")
        print(f"   - Correlación Pearson: {pearson_corr:.3f} (p-value: {pearson_p:.6f})")
        print(f"   - Correlación Spearman: {spearman_corr:.3f} (p-value: {spearman_p:.6f})")
        
        # Interpretación
        if abs(pearson_corr) > 0.7:
            interpretation = "FUERTE"
        elif abs(pearson_corr) > 0.5:
            interpretation = "MODERADA"
        elif abs(pearson_corr) > 0.3:
            interpretation = "DÉBIL"
        else:
            interpretation = "MUY DÉBIL"
        
        print(f"   - Interpretación: Correlación {interpretation}")
        
        # Análisis de discrepancias
        differences = np.array(estimated_quality) - np.array(original_quality)
        print(f"\n📊 ANÁLISIS DE DISCREPANCIAS:")
        print(f"   - Diferencia promedio: {np.mean(differences):.1f}")
        print(f"   - Desviación estándar: {np.std(differences):.1f}")
        print(f"   - MAE (Error Absoluto Medio): {np.mean(np.abs(differences)):.1f}")
        print(f"   - RMSE (Error Cuadrático Medio): {np.sqrt(np.mean(differences**2)):.1f}")
        
        # Identificar outliers (diferencias extremas)
        outlier_threshold = np.std(differences) * 2
        outliers = [(node_ids[i], estimated_quality[i], original_quality[i], differences[i]) 
                   for i, diff in enumerate(differences) 
                   if abs(diff) > outlier_threshold]
        
        print(f"\n🚨 OUTLIERS IDENTIFICADOS ({len(outliers)} nodos):")
        for node_id, est, orig, diff in outliers[:10]:  # Mostrar solo los primeros 10
            print(f"   {node_id}: Estimada={est:.1f}, Original={orig:.1f}, Diff={diff:.1f}")
        
        # Guardar resultados
        self.correlation_results = {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'interpretation': interpretation,
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            'mae': np.mean(np.abs(differences)),
            'rmse': np.sqrt(np.mean(differences**2)),
            'outliers_count': len(outliers),
            'common_nodes_count': len(common_nodes)
        }
        
        return {
            'estimated_quality': estimated_quality,
            'original_quality': original_quality,
            'node_ids': node_ids,
            'differences': differences,
            'outliers': outliers
        }
    
    def create_correlation_visualizations(self, correlation_data):
        """Crear visualizaciones de correlación"""
        print("\n🎨 CREANDO VISUALIZACIONES DE CORRELACIÓN...")
        
        if correlation_data is None:
            print("⚠️ No hay datos de correlación para visualizar")
            return
        
        # Configurar figura
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ANÁLISIS DE CORRELACIÓN: CALIDAD ESTIMADA vs ORIGINAL', fontsize=16, fontweight='bold')
        
        estimated = correlation_data['estimated_quality']
        original = correlation_data['original_quality']
        differences = correlation_data['differences']
        
        # 1. Scatter plot de correlación
        ax1 = axes[0, 0]
        ax1.scatter(original, estimated, alpha=0.6, color='blue')
        ax1.plot([0, 100], [0, 100], 'r--', label='Línea perfecta (y=x)')
        
        # Línea de regresión
        z = np.polyfit(original, estimated, 1)
        p = np.poly1d(z)
        ax1.plot(original, p(original), "orange", linewidth=2, label=f'Regresión: y={z[0]:.2f}x+{z[1]:.1f}')
        
        ax1.set_xlabel('Calidad Original')
        ax1.set_ylabel('Calidad Estimada')
        ax1.set_title(f'Correlación Pearson: {self.correlation_results["pearson_correlation"]:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Histograma de diferencias
        ax2 = axes[0, 1]
        ax2.hist(differences, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(differences), color='red', linestyle='--', 
                   label=f'Media: {np.mean(differences):.1f}')
        ax2.set_xlabel('Diferencia (Estimada - Original)')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Diferencias')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot comparativo
        ax3 = axes[1, 0]
        ax3.boxplot([estimated, original], labels=['Estimada', 'Original'])
        ax3.set_ylabel('Calidad de Servicio')
        ax3.set_title('Comparación de Distribuciones')
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals plot
        ax4 = axes[1, 1]
        predicted = np.polyval(np.polyfit(original, estimated, 1), original)
        residuals = estimated - predicted
        ax4.scatter(predicted, residuals, alpha=0.6, color='purple')
        ax4.axhline(y=0, color='red', linestyle='--')
        ax4.set_xlabel('Valores Predichos')
        ax4.set_ylabel('Residuales')
        ax4.set_title('Análisis de Residuales')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quality_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualización guardada: quality_correlation_analysis.png")
    
    def generate_correlation_report(self):
        """Generar reporte completo de correlación"""
        print("\n=== GENERANDO REPORTE COMPLETO DE CORRELACIÓN ===")
        
        # Ejecutar análisis completo
        estimated_data = self.load_estimated_quality_data()
        original_quality_indicators = self.analyze_original_quality_indicators()
        original_scores = self.create_quality_scoring_from_original()
        correlation_data = self.correlate_estimated_vs_original(estimated_data, original_scores)
        
        # Crear visualizaciones
        self.create_correlation_visualizations(correlation_data)
        
        # Preparar reporte completo
        report = {
            'metadata': {
                'total_nodes_estimated': len(estimated_data),
                'total_nodes_original': len(original_scores),
                'common_nodes': len(set(estimated_data.keys()) & set(original_scores.keys())),
                'analysis_date': pd.Timestamp.now().isoformat()
            },
            'correlation_results': self.correlation_results,
            'estimated_stats': {
                'mean': np.mean([d['calidad_servicio_estimada'] for d in estimated_data.values()]),
                'std': np.std([d['calidad_servicio_estimada'] for d in estimated_data.values()]),
                'min': min([d['calidad_servicio_estimada'] for d in estimated_data.values()]),
                'max': max([d['calidad_servicio_estimada'] for d in estimated_data.values()])
            },
            'original_stats': {
                'mean': np.mean([d['calidad_original'] for d in original_scores.values()]),
                'std': np.std([d['calidad_original'] for d in original_scores.values()]),
                'min': min([d['calidad_original'] for d in original_scores.values()]),
                'max': max([d['calidad_original'] for d in original_scores.values()])
            }
        }
        
        # Guardar reporte
        with open('quality_correlation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ ANÁLISIS COMPLETADO:")
        print(f"   - quality_correlation_analysis.png: Visualizaciones")
        print(f"   - quality_correlation_report.json: Reporte detallado")
        
        # Resumen ejecutivo
        if self.correlation_results:
            print(f"\n📋 RESUMEN EJECUTIVO:")
            print(f"   - Correlación: {self.correlation_results['interpretation']}")
            print(f"   - Coeficiente Pearson: {self.correlation_results['pearson_correlation']:.3f}")
            print(f"   - Error Absoluto Medio: {self.correlation_results['mae']:.1f} puntos")
            print(f"   - Nodos analizados: {self.correlation_results['common_nodes_count']}")
        
        return report

def main():
    """Función principal"""
    analyzer = QualityCorrelationAnalysis()
    analyzer.generate_correlation_report()

if __name__ == "__main__":
    main()