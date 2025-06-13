#!/usr/bin/env python3
"""
Modelo ML Mejorado con Features Reales para Predicción de Calidad de Servicio
Enfoque: Usar más datos originales, menos estimaciones propias
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from scipy.stats import pearsonr, spearmanr
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from service_quality_matrix import ServiceQualityMatrix
from correlate_quality_analysis import QualityCorrelationAnalysis

class EnhancedMLPredictor:
    def __init__(self, csv_path="../public/Mediciones Originales CEB .csv"):
        """Inicializar predictor ML mejorado"""
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
        
    def create_enhanced_features(self):
        """Crear features mejorados enfocados en datos reales"""
        print("=== CREANDO FEATURES MEJORADOS (ENFOQUE DATOS REALES) ===\n")
        
        # 1. Cargar datos base
        estimated_data = self.sqm.create_nodes_analysis()
        super_nodes = self.sqm.create_super_nodes()
        original_scores = self.qca.create_quality_scoring_from_original()
        
        print(f"✅ Datos base cargados: {len(estimated_data)} nodos")
        
        # 2. Crear features expandidos por nodo
        node_features = {}
        
        # Crear diccionarios de agregación por subestación y alimentador
        subestacion_stats = self._calculate_subestacion_stats()
        alimentador_stats = self._calculate_alimentador_stats()
        
        processed_count = 0
        skipped_count = 0
        
        for node_id, data in estimated_data.items():
            if node_id in original_scores:
                # Buscar fila original
                try:
                    codigo_ct_str = node_id.replace('CT_', '')
                    # Verificar que sea un número válido
                    if not codigo_ct_str.replace('.', '').replace('-', '').isdigit():
                        skipped_count += 1
                        continue
                    
                    codigo_ct = int(float(codigo_ct_str))
                    matching_rows = self.df[self.df['Codigoct'] == codigo_ct]
                    original_row = matching_rows.iloc[0] if len(matching_rows) > 0 else None
                except (ValueError, TypeError, OverflowError):
                    # Si no podemos convertir, usar datos básicos sin features originales
                    original_row = None
                
                # No saltar si original_row es None, usar datos básicos
                processed_count += 1
                
                # === FEATURES BÁSICOS (SOLO DATOS REALES) ===
                features = {
                    # Datos básicos del transformador
                    'potencia_instalada': data['potencia_instalada'],
                    'usuarios': data['usuarios'],
                    'densidad_usuarios_real': data['usuarios'] / max(data['potencia_instalada'], 1),
                    
                    # Coordenadas y ubicación
                    'coordx': data['coordx'],
                    'coordy': data['coordy'],
                    'latitud_normalizada': (data['coordy'] + 41.13) * 100,
                    'longitud_normalizada': (data['coordx'] + 71.33) * 100,
                }
                
                # Agregar features de datos originales solo si están disponibles
                if original_row is not None:
                    features.update({
                        # === FEATURES DE INFRAESTRUCTURA REAL ===
                        'cantidad_transformadores': original_row.get('Cantidadtrafos', 1),
                        'usuarios_circuitos': original_row.get('Usuarios Circuitos', 0),
                        'tipo_instalacion_num': 1 if str(original_row.get('Tipoinstalacion', 'M')) == 'M' else 0,
                        
                        # === FEATURES TEMPORALES REALES ===
                        'tiene_fecha_inicio': 1 if pd.notna(original_row.get('Fechainicio')) else 0,
                        'año_instalacion': self._extract_year(original_row.get('Fechainicio')),
                        'antiguedad_años': 2025 - self._extract_year(original_row.get('Fechainicio')),
                        'es_instalacion_nueva': 1 if self._extract_year(original_row.get('Fechainicio')) >= 2020 else 0,
                        'es_instalacion_vieja': 1 if self._extract_year(original_row.get('Fechainicio')) <= 2010 else 0,
                        
                        # === FEATURES DE MEDICIÓN REALES ===
                        'tiene_medidor_epre': 1 if str(original_row.get('Nro_EPRE', 'Sin Medir')) != 'Sin Medir' else 0,
                        'resultado_exitoso': 1 if str(original_row.get('Resultado', 'Sin medicion')) == 'Correcta' else 0,
                        'resultado_fallido': 1 if str(original_row.get('Resultado', 'Sin medicion')) == 'Fallida' else 0,
                        'resultado_penalizado': 1 if str(original_row.get('Resultado', 'Sin medicion')) == 'Penalizada' else 0,
                        'sin_medicion': 1 if str(original_row.get('Resultado', 'Sin medicion')) == 'Sin medicion' else 0,
                        'tipo_medicion_num': self._normalize_tipo_medicion(original_row.get('Tipo de medicion', -1)),
                        
                        # === FEATURES DE CONTEXTO DE SUBESTACIÓN ===
                        'subestacion_id_hash': hash(str(original_row.get('Historial Ago 21- May 25.Subestacion', 'UNKNOWN'))) % 100,
                        'alimentador_id_hash': hash(str(original_row.get('Idalimentador', 'UNKNOWN'))) % 100,
                    })
                else:
                    # Valores por defecto cuando no hay datos originales
                    features.update({
                        'cantidad_transformadores': 1,
                        'usuarios_circuitos': 0,
                        'tipo_instalacion_num': 1,
                        'tiene_fecha_inicio': 0,
                        'año_instalacion': 2000,
                        'antiguedad_años': 25,
                        'es_instalacion_nueva': 0,
                        'es_instalacion_vieja': 0,
                        'tiene_medidor_epre': 0,
                        'resultado_exitoso': 0,
                        'resultado_fallido': 0,
                        'resultado_penalizado': 0,
                        'sin_medicion': 1,
                        'tipo_medicion_num': 0,
                        'subestacion_id_hash': 0,
                        'alimentador_id_hash': 0,
                    })
                
                # === FEATURES AGREGADOS POR SUBESTACIÓN ===
                if original_row is not None:
                    subestacion_name = str(original_row.get('Historial Ago 21- May 25.Subestacion', 'UNKNOWN'))
                    if subestacion_name in subestacion_stats:
                        sub_stats = subestacion_stats[subestacion_name]
                        features.update({
                            'sub_total_potencia': sub_stats['total_potencia'],
                            'sub_total_usuarios': sub_stats['total_usuarios'],
                            'sub_count_transformadores': sub_stats['count'],
                            'sub_promedio_potencia': sub_stats['promedio_potencia'],
                            'sub_std_potencia': sub_stats['std_potencia'],
                            'sub_promedio_usuarios': sub_stats['promedio_usuarios'],
                            'sub_densidad_promedio': sub_stats['densidad_promedio'],
                            'sub_exitosas_ratio': sub_stats['exitosas_ratio'],
                            'sub_fallidas_ratio': sub_stats['fallidas_ratio'],
                            'sub_con_epre_ratio': sub_stats['con_epre_ratio'],
                            
                            # Posición relativa dentro de la subestación
                            'posicion_potencia_en_sub': (data['potencia_instalada'] - sub_stats['promedio_potencia']) / max(sub_stats['std_potencia'], 1),
                            'posicion_usuarios_en_sub': (data['usuarios'] - sub_stats['promedio_usuarios']) / max(sub_stats['std_usuarios'], 1),
                            'es_transformador_grande_en_sub': 1 if data['potencia_instalada'] > sub_stats['promedio_potencia'] * 1.5 else 0,
                            'es_transformador_pequeño_en_sub': 1 if data['potencia_instalada'] < sub_stats['promedio_potencia'] * 0.5 else 0,
                        })
                    else:
                        # Valores por defecto para features de subestación
                        features.update({
                            'sub_total_potencia': 0,
                            'sub_total_usuarios': 0,
                            'sub_count_transformadores': 0,
                            'sub_promedio_potencia': 0,
                            'sub_std_potencia': 0,
                            'sub_promedio_usuarios': 0,
                            'sub_densidad_promedio': 0,
                            'sub_exitosas_ratio': 0,
                            'sub_fallidas_ratio': 0,
                            'sub_con_epre_ratio': 0,
                            'posicion_potencia_en_sub': 0,
                            'posicion_usuarios_en_sub': 0,
                            'es_transformador_grande_en_sub': 0,
                            'es_transformador_pequeño_en_sub': 0,
                        })
                else:
                    # Valores por defecto cuando no hay original_row
                    features.update({
                        'sub_total_potencia': 0,
                        'sub_total_usuarios': 0,
                        'sub_count_transformadores': 0,
                        'sub_promedio_potencia': 0,
                        'sub_std_potencia': 0,
                        'sub_promedio_usuarios': 0,
                        'sub_densidad_promedio': 0,
                        'sub_exitosas_ratio': 0,
                        'sub_fallidas_ratio': 0,
                        'sub_con_epre_ratio': 0,
                        'posicion_potencia_en_sub': 0,
                        'posicion_usuarios_en_sub': 0,
                        'es_transformador_grande_en_sub': 0,
                        'es_transformador_pequeño_en_sub': 0,
                    })
                
                # === FEATURES AGREGADOS POR ALIMENTADOR ===
                if original_row is not None:
                    alimentador_name = str(original_row.get('Idalimentador', 'UNKNOWN'))
                    if alimentador_name in alimentador_stats:
                        alim_stats = alimentador_stats[alimentador_name]
                        features.update({
                            'alim_total_potencia': alim_stats['total_potencia'],
                            'alim_total_usuarios': alim_stats['total_usuarios'],
                            'alim_count_transformadores': alim_stats['count'],
                            'alim_exitosas_ratio': alim_stats['exitosas_ratio'],
                            'alim_con_epre_ratio': alim_stats['con_epre_ratio'],
                            'alim_densidad_promedio': alim_stats['densidad_promedio'],
                        })
                    else:
                        # Valores por defecto para features de alimentador
                        features.update({
                            'alim_total_potencia': 0,
                            'alim_total_usuarios': 0,
                            'alim_count_transformadores': 0,
                            'alim_exitosas_ratio': 0,
                            'alim_con_epre_ratio': 0,
                            'alim_densidad_promedio': 0,
                        })
                else:
                    # Valores por defecto cuando no hay original_row
                    features.update({
                        'alim_total_potencia': 0,
                        'alim_total_usuarios': 0,
                        'alim_count_transformadores': 0,
                        'alim_exitosas_ratio': 0,
                        'alim_con_epre_ratio': 0,
                        'alim_densidad_promedio': 0,
                    })
                
                # === FEATURES GEOGRÁFICOS AVANZADOS ===
                features.update({
                    'distancia_centro_ciudad': np.sqrt((data['coordx'] + 71.33)**2 + (data['coordy'] + 41.13)**2),
                    'zona_geografica_x': 1 if data['coordx'] > -71.33 else 0,  # Este/Oeste
                    'zona_geografica_y': 1 if data['coordy'] > -41.13 else 0,  # Norte/Sur
                    'densidad_geografica_local': self._calculate_local_density(data['coordx'], data['coordy'], estimated_data),
                })
                
                # === FEATURES DE SUPER NODO (LIMITADOS) ===
                node_super_zone = None
                for zone_id, zone_data in super_nodes.items():
                    if node_id in zone_data['nodes']:
                        node_super_zone = zone_id
                        break
                
                if node_super_zone:
                    zone_data = super_nodes[node_super_zone]
                    features.update({
                        'super_nodo_id': int(zone_id.replace('ZONA_', '')),
                        'super_nodo_size': zone_data['nodes_count'],
                        'super_nodo_potencia_total': zone_data['total_potencia'],
                        'super_nodo_usuarios_total': zone_data['total_usuarios'],
                        'distancia_centro_super_nodo': np.sqrt(
                            (data['coordx'] - zone_data['coordx_centro'])**2 + 
                            (data['coordy'] - zone_data['coordy_centro'])**2
                        ),
                        'es_centro_super_nodo': 1 if abs(data['coordx'] - zone_data['coordx_centro']) < 0.01 and abs(data['coordy'] - zone_data['coordy_centro']) < 0.01 else 0,
                    })
                
                # === FEATURES DE INTERACCIÓN ===
                features.update({
                    'potencia_x_usuarios': data['potencia_instalada'] * data['usuarios'],
                    'densidad_x_antiguedad': features['densidad_usuarios_real'] * features['antiguedad_años'],
                    'medidor_x_potencia': features['tiene_medidor_epre'] * data['potencia_instalada'],
                    'exitoso_x_potencia': features['resultado_exitoso'] * data['potencia_instalada'],
                    'distancia_x_usuarios': features['distancia_centro_ciudad'] * data['usuarios'],
                })
                
                # Target: calidad original
                target_quality = original_scores[node_id]['calidad_original']
                
                node_features[node_id] = features
                self.target_data[node_id] = target_quality
        
        self.features_data = node_features
        print(f"✅ Features mejorados creados: {len(node_features)} nodos")
        print(f"📊 Features por nodo: {len(list(node_features.values())[0]) if node_features else 0}")
        
        # Mostrar categorías de features
        if node_features:
            sample_features = list(node_features.values())[0]
            categories = {
                'Infraestructura': [f for f in sample_features.keys() if any(x in f for x in ['potencia', 'usuarios', 'cantidad', 'tipo_instalacion'])],
                'Temporales': [f for f in sample_features.keys() if any(x in f for x in ['año', 'antiguedad', 'fecha', 'nuevo', 'viejo'])],
                'Medición': [f for f in sample_features.keys() if any(x in f for x in ['resultado', 'medidor', 'epre', 'medicion'])],
                'Geográficos': [f for f in sample_features.keys() if any(x in f for x in ['coord', 'distancia', 'zona', 'centro'])],
                'Agregados': [f for f in sample_features.keys() if any(x in f for x in ['sub_', 'alim_', 'super_nodo'])],
                'Interacción': [f for f in sample_features.keys() if '_x_' in f],
            }
            
            print(f"\n📋 CATEGORÍAS DE FEATURES:")
            for cat, feats in categories.items():
                print(f"   {cat}: {len(feats)} features")
        
        return node_features
    
    def _extract_year(self, fecha_str):
        """Extraer año de fecha string"""
        try:
            if pd.isna(fecha_str) or fecha_str == '':
                return 2000
            return int(str(fecha_str).split('-')[0])
        except:
            return 2000
    
    def _normalize_tipo_medicion(self, tipo):
        """Normalizar tipo de medición"""
        try:
            val = int(tipo)
            return max(0, min(val, 4))  # Entre 0 y 4
        except:
            return 0
    
    def _calculate_subestacion_stats(self):
        """Calcular estadísticas por subestación"""
        stats = {}
        for subestacion in self.df['Historial Ago 21- May 25.Subestacion'].unique():
            if pd.isna(subestacion):
                continue
            
            sub_data = self.df[self.df['Historial Ago 21- May 25.Subestacion'] == subestacion]
            sub_data = sub_data.dropna(subset=['POTENCIA', 'Usuarios Transformador'])
            
            if len(sub_data) == 0:
                continue
            
            stats[str(subestacion)] = {
                'total_potencia': sub_data['POTENCIA'].sum(),
                'total_usuarios': sub_data['Usuarios Transformador'].sum(),
                'count': len(sub_data),
                'promedio_potencia': sub_data['POTENCIA'].mean(),
                'std_potencia': sub_data['POTENCIA'].std() if len(sub_data) > 1 else 0,
                'promedio_usuarios': sub_data['Usuarios Transformador'].mean(),
                'std_usuarios': sub_data['Usuarios Transformador'].std() if len(sub_data) > 1 else 0,
                'densidad_promedio': (sub_data['Usuarios Transformador'] / sub_data['POTENCIA']).mean(),
                'exitosas_ratio': (sub_data['Resultado'] == 'Correcta').mean(),
                'fallidas_ratio': (sub_data['Resultado'] == 'Fallida').mean(),
                'con_epre_ratio': (sub_data['Nro_EPRE'] != 'Sin Medir').mean(),
            }
        
        return stats
    
    def _calculate_alimentador_stats(self):
        """Calcular estadísticas por alimentador"""
        stats = {}
        for alimentador in self.df['Idalimentador'].unique():
            if pd.isna(alimentador):
                continue
            
            alim_data = self.df[self.df['Idalimentador'] == alimentador]
            alim_data = alim_data.dropna(subset=['POTENCIA', 'Usuarios Transformador'])
            
            if len(alim_data) == 0:
                continue
            
            stats[str(alimentador)] = {
                'total_potencia': alim_data['POTENCIA'].sum(),
                'total_usuarios': alim_data['Usuarios Transformador'].sum(),
                'count': len(alim_data),
                'exitosas_ratio': (alim_data['Resultado'] == 'Correcta').mean(),
                'con_epre_ratio': (alim_data['Nro_EPRE'] != 'Sin Medir').mean(),
                'densidad_promedio': (alim_data['Usuarios Transformador'] / alim_data['POTENCIA']).mean(),
            }
        
        return stats
    
    def _calculate_local_density(self, x, y, all_nodes, radius=0.05):
        """Calcular densidad local de transformadores"""
        count = 0
        for node_data in all_nodes.values():
            dist = np.sqrt((node_data['coordx'] - x)**2 + (node_data['coordy'] - y)**2)
            if dist <= radius:
                count += 1
        return count
    
    def train_enhanced_models(self, features_df, targets):
        """Entrenar modelos ML mejorados"""
        print("\n=== ENTRENANDO MODELOS ML MEJORADOS ===\n")
        
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets, test_size=0.2, random_state=42
        )
        
        # Modelos mejorados
        models_to_test = {
            'random_forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'extra_trees': ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=10.0),
            'lasso': Lasso(alpha=1.0, max_iter=2000),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000),
            'knn': KNeighborsRegressor(n_neighbors=10),
            'svr_rbf': SVR(kernel='rbf', C=10.0, gamma='scale'),
        }
        
        # Escaladores
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'none': None
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
    
    def run_enhanced_analysis(self):
        """Ejecutar análisis ML mejorado"""
        print("=== ANÁLISIS ML MEJORADO PARA PREDICCIÓN DE CALIDAD ===\n")
        
        # 1. Crear features mejorados
        features_data = self.create_enhanced_features()
        
        # 2. Preparar datos
        features_df = pd.DataFrame.from_dict(features_data, orient='index')
        targets = pd.Series(self.target_data)
        
        # Limpiar datos
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.median())
        
        # Eliminar features constantes
        constant_features = features_df.columns[features_df.std() == 0]
        if len(constant_features) > 0:
            print(f"⚠️ Eliminando {len(constant_features)} features constantes")
            features_df = features_df.drop(columns=constant_features)
        
        print(f"✅ Features finales: {features_df.shape[1]} variables")
        print(f"📊 Target range: {targets.min():.1f} - {targets.max():.1f}")
        
        # 3. Entrenar modelos
        X_train, X_test, y_train, y_test, best_key = self.train_enhanced_models(features_df, targets)
        
        # 4. Analizar feature importance
        model = self.models[best_key]['model']
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 TOP 15 FEATURES MÁS IMPORTANTES:")
            for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
                print(f"   {i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")
        
        # 5. Guardar resultados
        enhanced_report = {
            'enhanced_model_validation': {
                'best_correlation': self.best_correlation,
                'improvement_vs_baseline': self.best_correlation - 0.138,  # vs correlación original
                'best_model_type': best_key,
                'is_simulation_ready': self.best_correlation >= 0.7,
                'total_features': features_df.shape[1],
                'training_samples': len(X_train)
            },
            'recommendations': self._generate_enhanced_recommendations()
        }
        
        with open('enhanced_ml_model.json', 'w') as f:
            json.dump(enhanced_report, f, indent=2, default=str)
        
        print(f"\n🎯 RESULTADOS MEJORADOS:")
        print(f"   📊 Correlación alcanzada: {self.best_correlation:.3f}")
        print(f"   📈 Mejora vs baseline (0.138): +{self.best_correlation - 0.138:.3f}")
        print(f"   📈 Mejora vs ML básico (0.343): +{self.best_correlation - 0.343:.3f}")
        print(f"   ✅ Listo para simulación: {'SÍ' if self.best_correlation >= 0.7 else 'NO'}")
        
        return enhanced_report
    
    def _generate_enhanced_recommendations(self):
        """Generar recomendaciones para el modelo mejorado"""
        recommendations = []
        
        if self.best_correlation >= 0.7:
            recommendations.append("EXCELENTE: Correlación fuerte alcanzada. Modelo listo para simulación de red.")
        elif self.best_correlation >= 0.6:
            recommendations.append("BUENO: Correlación moderada-alta. Modelo utilizable para simulación con precaución.")
        elif self.best_correlation >= 0.5:
            recommendations.append("ACEPTABLE: Correlación moderada. Necesario validar simulaciones con datos reales.")
        else:
            recommendations.append("INSUFICIENTE: Correlación baja. Requiere más datos o features diferentes.")
        
        if self.best_correlation > 0.343:
            recommendations.append(f"MEJORA CONFIRMADA: +{self.best_correlation - 0.343:.3f} vs modelo básico.")
        
        recommendations.append("Próximos pasos: Implementar simulación de caídas de frecuencia/potencia por super nodo.")
        
        return recommendations

def main():
    """Función principal"""
    predictor = EnhancedMLPredictor()
    predictor.run_enhanced_analysis()

if __name__ == "__main__":
    main()