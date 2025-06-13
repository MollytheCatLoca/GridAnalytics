#!/usr/bin/env python3
"""
Modelo de Machine Learning Jerárquico para Predicción de Calidad de Servicio
Objetivo: Encontrar parámetros críticos para simulación válida de la red eléctrica
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Importar nuestras clases existentes
from service_quality_matrix import ServiceQualityMatrix
from correlate_quality_analysis import QualityCorrelationAnalysis

class MLQualityPredictor:
    def __init__(self, csv_path="../public/Mediciones Originales CEB .csv"):
        """Inicializar predictor ML jerárquico"""
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.sqm = ServiceQualityMatrix(csv_path)
        self.qca = QualityCorrelationAnalysis(csv_path)
        
        # Datos procesados
        self.features_data = {}
        self.target_data = {}
        self.models = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_correlation = 0
        
    def create_hierarchical_features(self):
        """Crear features jerárquicos: nodo → super nodo → red"""
        print("=== CREANDO FEATURES JERÁRQUICOS ===\n")
        
        # 1. Cargar datos base
        estimated_data = self.sqm.create_nodes_analysis()
        super_nodes = self.sqm.create_super_nodes()
        original_scores = self.qca.create_quality_scoring_from_original()
        
        print(f"✅ Datos base cargados: {len(estimated_data)} nodos")
        
        # 2. Crear features por nodo
        node_features = {}
        for node_id, data in estimated_data.items():
            if node_id in original_scores:
                # Features básicos del nodo
                features = {
                    # === FEATURES TÉCNICOS ===
                    'potencia_instalada': data['potencia_instalada'],
                    'usuarios': data['usuarios'],
                    'densidad_usuarios': data['densidad_usuarios'],
                    'carga_especifica': data['carga_especifica'],
                    
                    # === FEATURES GEOGRÁFICOS ===
                    'coordx': data['coordx'],
                    'coordy': data['coordy'],
                    'distancia_centro': np.sqrt((data['coordx'] + 71.33)**2 + (data['coordy'] + 41.13)**2),
                    
                    # === FEATURES DE CARGA ===
                    'factor_sobrecarga': data['factor_sobrecarga'],
                    'factor_estabilidad': data['factor_estabilidad'],
                    'factor_confiabilidad': data['factor_confiabilidad'],
                    
                    # === FEATURES ESTIMADOS ORIGINALES ===
                    'fluctuacion_potencia_estimada': data['fluctuacion_potencia_estimada'],
                    'fluctuacion_tension_estimada': data['fluctuacion_tension_estimada'],
                    'calidad_servicio_estimada': data['calidad_servicio_estimada'],
                }
                
                # === FEATURES JERÁRQUICOS: SUPER NODO ===
                # Encontrar super nodo al que pertenece
                node_super_zone = None
                for zone_id, zone_data in super_nodes.items():
                    if node_id in zone_data['nodes']:
                        node_super_zone = zone_id
                        break
                
                if node_super_zone:
                    zone_data = super_nodes[node_super_zone]
                    features.update({
                        'super_nodo_criticidad': zone_data['criticidad_zona'],
                        'super_nodo_potencia_total': zone_data['total_potencia'],
                        'super_nodo_usuarios_total': zone_data['total_usuarios'],
                        'super_nodo_densidad_promedio': zone_data['densidad_promedio'],
                        'super_nodo_area': zone_data['area_geografica'],
                        'super_nodo_densidad_espacial': zone_data['densidad_espacial'],
                        'super_nodo_subestaciones': zone_data['subestaciones_en_zona'],
                        'super_nodo_nodes_count': zone_data['nodes_count'],
                        
                        # Posición relativa dentro del super nodo
                        'distancia_centro_super_nodo': np.sqrt(
                            (data['coordx'] - zone_data['coordx_centro'])**2 + 
                            (data['coordy'] - zone_data['coordy_centro'])**2
                        ),
                    })
                
                # === FEATURES CONTEXTUALES ===
                # Buscar datos adicionales del CSV original
                try:
                    codigo_ct = int(node_id.replace('CT_', ''))
                    matching_rows = self.df[self.df['Codigoct'] == codigo_ct]
                    original_row = matching_rows.iloc[0] if len(matching_rows) > 0 else None
                except (ValueError, TypeError):
                    original_row = None
                
                if original_row is not None:
                    # Features de la subestación
                    features.update({
                        'subestacion_id': hash(str(original_row.get('Historial Ago 21- May 25.Subestacion', 'UNKNOWN'))) % 1000,
                        'alimentador_id': hash(str(original_row.get('Idalimentador', 'UNKNOWN'))) % 1000,
                        'tipo_instalacion': 1 if str(original_row.get('Tipoinstalacion', 'M')) == 'M' else 0,
                        'cantidad_trafos': original_row.get('Cantidadtrafos', 1),
                        'usuarios_circuitos': original_row.get('Usuarios Circuitos', 0),
                        
                        # Features temporales
                        'tiene_fecha_inicio': 1 if pd.notna(original_row.get('Fechainicio')) else 0,
                        'año_instalacion': int(str(original_row.get('Fechainicio', '2000-01-01')).split('-')[0]) if pd.notna(original_row.get('Fechainicio')) else 2000,
                        'antiguedad': 2025 - int(str(original_row.get('Fechainicio', '2000-01-01')).split('-')[0]) if pd.notna(original_row.get('Fechainicio')) else 25,
                        
                        # Features de medición originales
                        'tiene_epre': 1 if str(original_row.get('Nro_EPRE', 'Sin Medir')) != 'Sin Medir' else 0,
                        'resultado_exitoso': 1 if str(original_row.get('Resultado', 'Sin medicion')) == 'Exitosa' else 0,
                        'resultado_fallido': 1 if str(original_row.get('Resultado', 'Sin medicion')) == 'Fallida' else 0,
                        'resultado_penalizado': 1 if str(original_row.get('Resultado', 'Sin medicion')) == 'Penalizada' else 0,
                        'tipo_medicion': original_row.get('Tipo de medicion', -1),
                    })
                
                # === FEATURES ESTADÍSTICOS POR ZONA ===
                # Calcular estadísticas de nodos vecinos en la misma zona
                if node_super_zone:
                    zone_nodes_data = [estimated_data[nid] for nid in zone_data['nodes'] if nid in estimated_data]
                    if len(zone_nodes_data) > 1:
                        potencias = [nd['potencia_instalada'] for nd in zone_nodes_data]
                        usuarios_list = [nd['usuarios'] for nd in zone_nodes_data]
                        
                        features.update({
                            'zona_potencia_percentil': np.percentile(potencias, 50) if len(potencias) > 0 else 0,
                            'zona_usuarios_percentil': np.percentile(usuarios_list, 50) if len(usuarios_list) > 0 else 0,
                            'zona_potencia_std': np.std(potencias) if len(potencias) > 1 else 0,
                            'zona_usuarios_std': np.std(usuarios_list) if len(usuarios_list) > 1 else 0,
                            'posicion_potencia_en_zona': (data['potencia_instalada'] - np.mean(potencias)) / (np.std(potencias) + 0.001) if len(potencias) > 1 else 0,
                            'posicion_usuarios_en_zona': (data['usuarios'] - np.mean(usuarios_list)) / (np.std(usuarios_list) + 0.001) if len(usuarios_list) > 1 else 0,
                        })
                
                # Target: calidad original
                target_quality = original_scores[node_id]['calidad_original']
                
                node_features[node_id] = features
                self.target_data[node_id] = target_quality
        
        self.features_data = node_features
        print(f"✅ Features jerárquicos creados: {len(node_features)} nodos")
        print(f"📊 Features por nodo: {len(list(node_features.values())[0]) if node_features else 0}")
        
        # Mostrar ejemplos de features
        if node_features:
            sample_node = list(node_features.keys())[0]
            print(f"\n📋 EJEMPLO DE FEATURES ({sample_node}):")
            for key, value in list(node_features[sample_node].items())[:10]:
                print(f"   {key}: {value}")
            print("   ...")
        
        return node_features
    
    def prepare_ml_data(self):
        """Preparar datos para machine learning"""
        print("\n=== PREPARANDO DATOS PARA ML ===\n")
        
        if not self.features_data:
            self.create_hierarchical_features()
        
        # Convertir a DataFrame
        features_df = pd.DataFrame.from_dict(self.features_data, orient='index')
        targets = pd.Series(self.target_data)
        
        # Verificar alineación
        common_indices = features_df.index.intersection(targets.index)
        features_df = features_df.loc[common_indices]
        targets = targets.loc[common_indices]
        
        print(f"✅ Datos alineados: {len(features_df)} muestras")
        print(f"✅ Features: {features_df.shape[1]} variables")
        
        # Limpiar datos
        # Reemplazar infinitos y NaN
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.median())
        
        # Eliminar features constantes
        constant_features = features_df.columns[features_df.std() == 0]
        if len(constant_features) > 0:
            print(f"⚠️ Eliminando {len(constant_features)} features constantes")
            features_df = features_df.drop(columns=constant_features)
        
        print(f"✅ Features finales: {features_df.shape[1]} variables")
        print(f"📊 Target range: {targets.min():.1f} - {targets.max():.1f}")
        
        return features_df, targets
    
    def train_multiple_models(self, features_df, targets):
        """Entrenar múltiples modelos ML"""
        print("\n=== ENTRENANDO MÚLTIPLES MODELOS ML ===\n")
        
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets, test_size=0.2, random_state=42
        )
        
        # Escaladores
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'none': None
        }
        
        # Modelos a probar
        models_to_test = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'svr': SVR(kernel='rbf', C=1.0),
        }
        
        results = {}
        
        # Probar todas las combinaciones
        for scaler_name, scaler in scalers.items():
            for model_name, model in models_to_test.items():
                try:
                    print(f"🔄 Entrenando {model_name} con scaler {scaler_name}...")
                    
                    # Preparar datos
                    if scaler is not None:
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                    else:
                        X_train_scaled = X_train
                        X_test_scaled = X_test
                    
                    # Entrenar modelo
                    model.fit(X_train_scaled, y_train)
                    
                    # Predicciones
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    # Métricas
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    train_mae = mean_absolute_error(y_train, y_pred_train)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    
                    # Correlación
                    corr_train, _ = pearsonr(y_train, y_pred_train)
                    corr_test, _ = pearsonr(y_test, y_pred_test)
                    
                    # Validación cruzada
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    
                    key = f"{model_name}_{scaler_name}"
                    results[key] = {
                        'model': model,
                        'scaler': scaler,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'train_correlation': corr_train,
                        'test_correlation': corr_test,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'overfitting': abs(train_r2 - test_r2)
                    }
                    
                    print(f"   ✅ R²: {test_r2:.3f} | Correlación: {corr_test:.3f} | CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    continue
        
        # Encontrar mejor modelo
        self.models = results
        best_key = max(results.keys(), key=lambda k: results[k]['test_correlation'])
        self.best_model = results[best_key]
        self.best_correlation = self.best_model['test_correlation']
        
        print(f"\n🏆 MEJOR MODELO: {best_key}")
        print(f"   📊 Correlación Test: {self.best_correlation:.3f}")
        print(f"   📊 R² Test: {self.best_model['test_r2']:.3f}")
        print(f"   📊 MAE Test: {self.best_model['test_mae']:.1f}")
        print(f"   📊 Overfitting: {self.best_model['overfitting']:.3f}")
        
        return X_train, X_test, y_train, y_test, best_key
    
    def analyze_feature_importance(self, X_train, best_model_key):
        """Analizar importancia de features"""
        print("\n=== ANÁLISIS DE IMPORTANCIA DE FEATURES ===\n")
        
        model = self.models[best_model_key]['model']
        feature_names = X_train.columns
        
        # Obtener importancias según el tipo de modelo
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print("⚠️ Modelo no soporta análisis de importancia")
            return None
        
        # Crear DataFrame de importancias
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        print("🔝 TOP 20 FEATURES MÁS IMPORTANTES:")
        for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")
        
        # Agrupar por categorías
        categories = {
            'TÉCNICOS': ['potencia_instalada', 'usuarios', 'densidad_usuarios', 'carga_especifica', 'factor_sobrecarga'],
            'GEOGRÁFICOS': ['coordx', 'coordy', 'distancia_centro', 'distancia_centro_super_nodo'],
            'SUPER_NODO': [col for col in feature_names if col.startswith('super_nodo_')],
            'ORIGINALES': ['resultado_exitoso', 'resultado_fallido', 'tiene_epre', 'tipo_medicion'],
            'TEMPORALES': ['antiguedad', 'año_instalacion', 'tiene_fecha_inicio'],
            'CONTEXTUALES': ['subestacion_id', 'alimentador_id', 'cantidad_trafos'],
            'ESTIMADOS': ['fluctuacion_potencia_estimada', 'fluctuacion_tension_estimada', 'calidad_servicio_estimada']
        }
        
        print(f"\n📊 IMPORTANCIA POR CATEGORÍAS:")
        category_importance = {}
        for category, features in categories.items():
            cat_features = [f for f in features if f in feature_names]
            if cat_features:
                cat_importance = importance_df[importance_df['feature'].isin(cat_features)]['importance'].sum()
                category_importance[category] = cat_importance
                print(f"   {category:<15}: {cat_importance:.4f}")
        
        return importance_df, category_importance
    
    def create_visualizations(self, X_test, y_test, best_model_key):
        """Crear visualizaciones del modelo"""
        print("\n🎨 CREANDO VISUALIZACIONES...")
        
        # Configurar figura
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'ANÁLISIS ML - PREDICCIÓN CALIDAD DE SERVICIO\nMejor Modelo: {best_model_key}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Comparación de modelos
        ax1 = axes[0, 0]
        model_names = []
        correlations = []
        r2_scores = []
        
        for key, result in self.models.items():
            model_names.append(key.replace('_', '\n'))
            correlations.append(result['test_correlation'])
            r2_scores.append(result['test_r2'])
        
        x_pos = np.arange(len(model_names))
        ax1.bar(x_pos, correlations, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Modelos')
        ax1.set_ylabel('Correlación Test')
        ax1.set_title('Comparación de Modelos')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Predicciones vs Real
        ax2 = axes[0, 1]
        model = self.models[best_model_key]['model']
        scaler = self.models[best_model_key]['scaler']
        
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
            
        y_pred = model.predict(X_test_scaled)
        
        ax2.scatter(y_test, y_pred, alpha=0.6, color='blue')
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax2.set_xlabel('Calidad Real')
        ax2.set_ylabel('Calidad Predicha')
        ax2.set_title(f'Predicciones vs Real\nCorrelación: {self.best_correlation:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuales
        ax3 = axes[0, 2]
        residuals = y_test - y_pred
        ax3.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax3.axhline(y=0, color='red', linestyle='--')
        ax3.set_xlabel('Predicciones')
        ax3.set_ylabel('Residuales')
        ax3.set_title('Análisis de Residuales')
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature Importance
        ax4 = axes[1, 0]
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(15)
            ax4.barh(range(len(top_features)), top_features['importance'], color='coral')
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels(top_features['feature'], fontsize=8)
            ax4.set_xlabel('Importancia')
            ax4.set_title('Top 15 Features Importantes')
            ax4.grid(True, alpha=0.3)
        
        # 5. Distribución de errores
        ax5 = axes[1, 1]
        ax5.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax5.axvline(residuals.mean(), color='red', linestyle='--', 
                   label=f'Media: {residuals.mean():.1f}')
        ax5.set_xlabel('Residuales')
        ax5.set_ylabel('Frecuencia')
        ax5.set_title('Distribución de Errores')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Learning curves (aproximado)
        ax6 = axes[1, 2]
        model_scores = []
        model_names_short = []
        for key, result in self.models.items():
            if len(model_names_short) < 10:  # Limitar para visualización
                model_names_short.append(key.split('_')[0])
                model_scores.append([result['train_correlation'], result['test_correlation']])
        
        if model_scores:
            train_scores = [s[0] for s in model_scores]
            test_scores = [s[1] for s in model_scores]
            
            x_pos = np.arange(len(model_names_short))
            width = 0.35
            
            ax6.bar(x_pos - width/2, train_scores, width, label='Train', alpha=0.7, color='lightblue')
            ax6.bar(x_pos + width/2, test_scores, width, label='Test', alpha=0.7, color='lightcoral')
            
            ax6.set_xlabel('Modelos')
            ax6.set_ylabel('Correlación')
            ax6.set_title('Train vs Test Performance')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(model_names_short, rotation=45, ha='right', fontsize=8)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ml_quality_predictor_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualización guardada: ml_quality_predictor_analysis.png")
    
    def generate_simulation_model(self):
        """Generar modelo de simulación basado en ML validado"""
        print("\n=== GENERANDO MODELO DE SIMULACIÓN ===\n")
        
        if self.best_correlation < 0.5:
            print(f"⚠️ Correlación baja ({self.best_correlation:.3f}). Modelo puede no ser confiable para simulación.")
            print("   Recomendación: Mejorar features o conseguir más datos.")
        
        # Crear reporte de simulación
        simulation_report = {
            'model_validation': {
                'best_correlation': self.best_correlation,
                'best_model_type': max(self.models.keys(), key=lambda k: self.models[k]['test_correlation']),
                'model_performance': self.best_model,
                'is_simulation_ready': self.best_correlation >= 0.7
            },
            'critical_parameters': self.feature_importance.head(10).to_dict('records') if self.feature_importance is not None else [],
            'simulation_recommendations': self._generate_simulation_recommendations()
        }
        
        # Guardar modelo para simulación
        with open('ml_simulation_model.json', 'w') as f:
            json.dump(simulation_report, f, indent=2, default=str)
        
        print(f"✅ Modelo de simulación generado:")
        print(f"   - Correlación alcanzada: {self.best_correlation:.3f}")
        print(f"   - ¿Listo para simulación?: {'SÍ' if self.best_correlation >= 0.7 else 'NO'}")
        print(f"   - Archivo: ml_simulation_model.json")
        
        return simulation_report
    
    def _generate_simulation_recommendations(self):
        """Generar recomendaciones para simulación"""
        recommendations = []
        
        if self.best_correlation < 0.3:
            recommendations.append("CRÍTICO: Correlación muy baja. Revisar features y calidad de datos.")
        elif self.best_correlation < 0.5:
            recommendations.append("MEDIO: Correlación baja. Agregar más features contextuales.")
        elif self.best_correlation < 0.7:
            recommendations.append("BUENO: Correlación moderada. Modelo utilizable con precaución.")
        else:
            recommendations.append("EXCELENTE: Correlación alta. Modelo listo para simulación.")
        
        if self.feature_importance is not None:
            top_feature = self.feature_importance.iloc[0]['feature']
            recommendations.append(f"Feature más importante: {top_feature}")
            
            # Analizar categorías importantes
            if any('super_nodo' in f for f in self.feature_importance.head(5)['feature']):
                recommendations.append("Los super nodos son críticos para la predicción.")
            
            if any('resultado' in f for f in self.feature_importance.head(5)['feature']):
                recommendations.append("Los resultados de medición son factores clave.")
        
        return recommendations
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("=== ANÁLISIS ML COMPLETO PARA PREDICCIÓN DE CALIDAD ===\n")
        
        # 1. Crear features
        features_df, targets = self.prepare_ml_data()
        
        # 2. Entrenar modelos
        X_train, X_test, y_train, y_test, best_key = self.train_multiple_models(features_df, targets)
        
        # 3. Analizar importancia
        importance_df, category_importance = self.analyze_feature_importance(X_train, best_key)
        
        # 4. Crear visualizaciones
        self.create_visualizations(X_test, y_test, best_key)
        
        # 5. Generar modelo de simulación
        simulation_report = self.generate_simulation_model()
        
        print(f"\n🎯 OBJETIVO ALCANZADO:")
        if self.best_correlation >= 0.7:
            print(f"   ✅ CORRELACIÓN FUERTE: {self.best_correlation:.3f}")
            print(f"   ✅ Modelo listo para simular caídas de frecuencia/potencia")
            print(f"   ✅ Parámetros críticos identificados")
        else:
            print(f"   ⚠️ CORRELACIÓN: {self.best_correlation:.3f} (objetivo: >0.7)")
            print(f"   📋 Necesario mejorar features o datos para simulación válida")
        
        return simulation_report

def main():
    """Función principal"""
    predictor = MLQualityPredictor()
    predictor.run_complete_analysis()

if __name__ == "__main__":
    main()