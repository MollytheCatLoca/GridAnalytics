#!/usr/bin/env python3
"""
Script de prueba para el análisis de red CEB
"""

import pandas as pd
import json
from backend.main import get_grid_stats, get_pain_points

async def test_analysis():
    """Prueba el análisis de red"""
    print("=== PRUEBA DE ANÁLISIS DE RED CEB ===\n")
    
    try:
        # Probar estadísticas
        print("1. Obteniendo estadísticas de red...")
        stats = await get_grid_stats()
        print(f"✅ Estadísticas obtenidas:")
        print(f"   - Subestaciones: {stats.total_subestaciones}")
        print(f"   - Potencia total: {stats.potencia_total/1000:.1f} MW")
        print(f"   - Usuarios totales: {stats.usuarios_total:,}")
        print(f"   - Área geográfica: ({stats.area_geografica['center_x']:.2f}, {stats.area_geografica['center_y']:.2f})")
        
        # Probar puntos de dolor
        print("\n2. Identificando puntos de dolor...")
        pain_points = await get_pain_points()
        print(f"✅ Puntos de dolor identificados: {len(pain_points)}")
        
        if pain_points:
            print("\n   Top 5 puntos críticos:")
            for i, point in enumerate(pain_points[:5], 1):
                print(f"   {i}. {point.tipo} - {point.severidad} ({point.usuarios_afectados} usuarios, {point.potencia_afectada:.1f} kVA)")
                print(f"      💡 {point.recomendacion}")
        
        print(f"\n=== RESUMEN EJECUTIVO ===")
        print(f"La red CEB tiene {stats.total_subestaciones} subestaciones con {stats.potencia_total/1000:.1f} MW")
        print(f"instalados sirviendo a {stats.usuarios_total:,} usuarios.")
        print(f"Se identificaron {len(pain_points)} puntos de dolor que podrían beneficiarse")
        print(f"de parques solares para estabilización y optimización de la red.")
        
        # Calcular potencial solar
        alta_severidad = [p for p in pain_points if p.severidad == 'alta']
        media_severidad = [p for p in pain_points if p.severidad == 'media']
        
        potencia_alta = sum(p.potencia_afectada for p in alta_severidad)
        potencia_media = sum(p.potencia_afectada for p in media_severidad)
        
        print(f"\n=== OPORTUNIDADES DE ENERGÍA SOLAR ===")
        print(f"Puntos de alta severidad: {len(alta_severidad)} ({potencia_alta:.1f} kVA afectados)")
        print(f"Puntos de media severidad: {len(media_severidad)} ({potencia_media:.1f} kVA afectados)")
        print(f"Potencia total candidata para parques solares: {potencia_alta + potencia_media:.1f} kVA")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_analysis())
    print(f"\n✅ Análisis completado {'exitosamente' if success else 'con errores'}")