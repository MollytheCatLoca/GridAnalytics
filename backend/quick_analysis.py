#!/usr/bin/env python3
"""
Análisis rápido de la red CEB sin dependencias externas
"""

import pandas as pd
import json

def analyze_ceb_grid():
    """Analiza la red CEB y genera reporte"""
    print("=== ANÁLISIS DE RED CEB - COOPERATIVA ELÉCTRICA DE BOLÍVAR ===\n")
    
    try:
        # Cargar datos
        df = pd.read_csv("../public/Mediciones Originales CEB .csv")
        print(f"✅ Datos cargados: {len(df)} registros")
        
        # Estadísticas básicas
        print("\n=== ESTADÍSTICAS GENERALES ===")
        subestaciones = len(df['Historial Ago 21- May 25.Subestacion'].dropna().unique())
        alimentadores = len(df['Idalimentador'].dropna().unique())
        potencia_total = df['POTENCIA'].sum()
        usuarios_total = df['Usuarios Transformador'].sum()
        
        print(f"📍 Subestaciones: {subestaciones}")
        print(f"⚡ Alimentadores: {alimentadores}")
        print(f"🔋 Potencia total: {potencia_total/1000:.1f} MW")
        print(f"👥 Usuarios totales: {usuarios_total:,}")
        
        # Análisis geográfico
        coords_validas = df.dropna(subset=['Coordx', 'Coordy'])
        print(f"📍 Registros con coordenadas: {len(coords_validas)}")
        centro_x = coords_validas['Coordx'].mean()
        centro_y = coords_validas['Coordy'].mean()
        print(f"🌍 Centro geográfico: ({centro_x:.3f}, {centro_y:.3f})")
        
        # Identificar puntos de dolor
        print("\n=== ANÁLISIS DE PUNTOS DE DOLOR ===")
        
        dolor_sobrecarga = []
        dolor_baja_potencia = []
        
        # Sobrecarga por densidad usuarios/potencia
        for _, row in df.dropna(subset=['Usuarios Transformador', 'POTENCIA', 'Coordx', 'Coordy']).iterrows():
            if row['POTENCIA'] > 0:
                ratio = row['Usuarios Transformador'] / row['POTENCIA']
                if ratio > 1.0:  # Más de 1 usuario por kVA
                    severidad = "ALTA" if ratio > 1.5 else "MEDIA"
                    dolor_sobrecarga.append({
                        'circuito': row['Codigoct'],
                        'tipo': 'SOBRECARGA',
                        'ratio': ratio,
                        'severidad': severidad,
                        'usuarios': row['Usuarios Transformador'],
                        'potencia': row['POTENCIA'],
                        'coords': (row['Coordx'], row['Coordy'])
                    })
        
        # Baja potencia con muchos usuarios
        potencia_media = df['POTENCIA'].mean()
        for _, row in df.dropna(subset=['Usuarios Transformador', 'POTENCIA', 'Coordx', 'Coordy']).iterrows():
            if row['POTENCIA'] < potencia_media * 0.5 and row['Usuarios Transformador'] > 50:
                dolor_baja_potencia.append({
                    'circuito': row['Codigoct'],
                    'tipo': 'BAJA_POTENCIA',
                    'usuarios': row['Usuarios Transformador'],
                    'potencia': row['POTENCIA'],
                    'severidad': 'MEDIA',
                    'coords': (row['Coordx'], row['Coordy'])
                })
        
        total_dolor = len(dolor_sobrecarga) + len(dolor_baja_potencia)
        print(f"🚨 Total puntos de dolor identificados: {total_dolor}")
        print(f"   - Sobrecarga: {len(dolor_sobrecarga)}")
        print(f"   - Baja potencia: {len(dolor_baja_potencia)}")
        
        # Top 10 puntos críticos
        todos_puntos = dolor_sobrecarga + dolor_baja_potencia
        todos_puntos.sort(key=lambda x: x['usuarios'], reverse=True)
        
        print(f"\n=== TOP 10 PUNTOS MÁS CRÍTICOS ===")
        for i, punto in enumerate(todos_puntos[:10], 1):
            print(f"{i:2d}. CT-{punto['circuito']} | {punto['tipo']} | {punto['severidad']}")
            print(f"     👥 {punto['usuarios']} usuarios | ⚡ {punto['potencia']:.1f} kVA")
            if punto['tipo'] == 'SOBRECARGA':
                print(f"     📊 Ratio: {punto['ratio']:.2f} usuarios/kVA")
            print(f"     📍 Coordenadas: ({punto['coords'][0]:.3f}, {punto['coords'][1]:.3f})")
            print()
        
        # Análisis de oportunidades solares
        print("=== OPORTUNIDADES DE ENERGÍA SOLAR ===")
        
        puntos_alta = [p for p in todos_puntos if p['severidad'] == 'ALTA']
        puntos_media = [p for p in todos_puntos if p['severidad'] == 'MEDIA']
        
        potencia_alta = sum(p['potencia'] for p in puntos_alta)
        potencia_media = sum(p['potencia'] for p in puntos_media)
        usuarios_alta = sum(p['usuarios'] for p in puntos_alta)
        usuarios_media = sum(p['usuarios'] for p in puntos_media)
        
        print(f"🔴 Severidad ALTA: {len(puntos_alta)} puntos")
        print(f"   - Potencia afectada: {potencia_alta:.1f} kVA")
        print(f"   - Usuarios afectados: {usuarios_alta:,}")
        print(f"   - Recomendación: Parques solares de 500-1000 kW")
        
        print(f"🟡 Severidad MEDIA: {len(puntos_media)} puntos")
        print(f"   - Potencia afectada: {potencia_media:.1f} kVA")
        print(f"   - Usuarios afectados: {usuarios_media:,}")
        print(f"   - Recomendación: Parques solares de 200-500 kW")
        
        # Estimación de capacidad solar requerida
        capacidad_solar_recomendada = (potencia_alta * 0.8) + (potencia_media * 0.5)
        print(f"\n💡 CAPACIDAD SOLAR RECOMENDADA TOTAL: {capacidad_solar_recomendada/1000:.1f} MW")
        print(f"💰 Inversión estimada: ${capacidad_solar_recomendada * 1200:,.0f} USD")
        print(f"🌱 Reducción CO2 anual estimada: {capacidad_solar_recomendada * 1.5:.0f} toneladas")
        
        # Resumen ejecutivo
        print(f"\n=== RESUMEN EJECUTIVO ===")
        print(f"La red de CEB opera con {subestaciones} subestaciones distribuyendo {potencia_total/1000:.1f} MW")
        print(f"a {usuarios_total:,} usuarios a través de {alimentadores} alimentadores.")
        print(f"Se identificaron {total_dolor} puntos críticos que representan oportunidades")
        print(f"de mejora mediante la instalación de {capacidad_solar_recomendada/1000:.1f} MW de energía solar.")
        print(f"Esta inversión estabilizaría la red, reduciría costos operativos y mejoraría")
        print(f"la calidad del servicio para más de {usuarios_alta + usuarios_media:,} usuarios.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = analyze_ceb_grid()
    print(f"\n{'='*60}")
    print(f"✅ Análisis completado {'exitosamente' if success else 'con errores'}")
    print(f"{'='*60}")