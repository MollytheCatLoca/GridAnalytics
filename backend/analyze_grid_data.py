import pandas as pd
import numpy as np
from collections import defaultdict
import json

def analyze_grid_topology():
    """
    Analiza la estructura de datos para reconstruir la topología de la red eléctrica CEB
    """
    # Cargar datos
    df = pd.read_csv("public/Mediciones Originales CEB .csv")
    
    print("=== ANÁLISIS DE TOPOLOGÍA DE RED CEB ===\n")
    
    # 1. Información general
    print(f"Total de registros: {len(df)}")
    print(f"Columnas disponibles: {len(df.columns)}")
    
    # 2. Análisis de jerarquía de red
    print("\n=== JERARQUÍA DE RED ===")
    
    # Subestaciones
    subestaciones = df['Historial Ago 21- May 25.Subestacion'].dropna().unique()
    print(f"Subestaciones: {len(subestaciones)}")
    print(f"Lista de subestaciones: {list(subestaciones)[:10]}...")
    
    # Alimentadores
    alimentadores = df['Idalimentador'].dropna().unique()
    alimentadores_mt = df['Idalimentadormt'].dropna().unique()
    print(f"Alimentadores: {len(alimentadores)}")
    print(f"Alimentadores MT: {len(alimentadores_mt)}")
    
    # Transformadores
    transformadores = df['Idtransformador'].dropna().unique()
    print(f"Transformadores: {len(transformadores)}")
    
    # Circuitos
    circuitos = df['Codigoct'].dropna().unique()
    print(f"Circuitos: {len(circuitos)}")
    
    # 3. Análisis geográfico
    print("\n=== ANÁLISIS GEOGRÁFICO ===")
    coords_validas = df.dropna(subset=['Coordx', 'Coordy'])
    print(f"Registros con coordenadas: {len(coords_validas)}")
    
    if len(coords_validas) > 0:
        print(f"Rango X: {coords_validas['Coordx'].min():.2f} - {coords_validas['Coordx'].max():.2f}")
        print(f"Rango Y: {coords_validas['Coordy'].min():.2f} - {coords_validas['Coordy'].max():.2f}")
        print(f"Centro aproximado: ({coords_validas['Coordx'].mean():.2f}, {coords_validas['Coordy'].mean():.2f})")
    
    # 4. Análisis de potencia
    print("\n=== ANÁLISIS DE POTENCIA ===")
    potencia_data = df['POTENCIA'].dropna()
    potencia_hist = df['Historial Ago 21- May 25.Potencia'].dropna()
    
    print(f"Registros con potencia: {len(potencia_data)}")
    if len(potencia_data) > 0:
        print(f"Potencia total instalada: {potencia_data.sum():.2f} kVA")
        print(f"Potencia promedio por transformador: {potencia_data.mean():.2f} kVA")
        print(f"Rango de potencias: {potencia_data.min()} - {potencia_data.max()} kVA")
    
    # 5. Análisis de usuarios
    print("\n=== ANÁLISIS DE USUARIOS ===")
    usuarios_circuito = df['Usuarios Circuitos'].dropna()
    usuarios_trafo = df['Usuarios Transformador'].dropna()
    
    print(f"Total usuarios por circuitos: {usuarios_circuito.sum()}")
    print(f"Total usuarios por transformadores: {usuarios_trafo.sum()}")
    print(f"Promedio usuarios por circuito: {usuarios_circuito.mean():.2f}")
    print(f"Promedio usuarios por transformador: {usuarios_trafo.mean():.2f}")
    
    # 6. Reconstruir jerarquía
    print("\n=== RECONSTRUCCIÓN DE JERARQUÍA ===")
    
    # Crear estructura jerárquica
    grid_structure = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for _, row in df.iterrows():
        subestacion = row.get('Historial Ago 21- May 25.Subestacion', 'DESCONOCIDA')
        alimentador = row.get('Idalimentador', 'DESCONOCIDO')
        transformador = row.get('Idtransformador', 'DESCONOCIDO')
        circuito = row.get('Codigoct', 'DESCONOCIDO')
        
        if pd.notna(subestacion) and pd.notna(alimentador) and pd.notna(transformador):
            grid_structure[str(subestacion)][str(alimentador)][str(transformador)].append({
                'circuito': circuito,
                'coordx': row.get('Coordx'),
                'coordy': row.get('Coordy'),
                'potencia': row.get('POTENCIA'),
                'usuarios': row.get('Usuarios Transformador'),
                'direccion': row.get('Direccion')
            })
    
    # Mostrar estructura
    for subestacion, alimentadores in list(grid_structure.items())[:3]:
        print(f"\nSubestación: {subestacion}")
        for alimentador, transformadores in list(alimentadores.items())[:2]:
            print(f"  └─ Alimentador: {alimentador} ({len(transformadores)} transformadores)")
            for transformador, circuitos in list(transformadores.items())[:3]:
                print(f"      └─ Transformador: {transformador} ({len(circuitos)} circuitos)")
    
    # Guardar estructura en JSON
    # Convertir a formato serializable
    serializable_structure = {}
    for sub, alims in grid_structure.items():
        serializable_structure[sub] = {}
        for alim, trafos in alims.items():
            serializable_structure[sub][alim] = {}
            for trafo, circuits in trafos.items():
                serializable_structure[sub][alim][trafo] = circuits
    
    with open('grid_topology.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_structure, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ Estructura guardada en 'grid_topology.json'")
    
    # 7. Estadísticas finales
    print("\n=== ESTADÍSTICAS FINALES ===")
    print(f"Subestaciones identificadas: {len(grid_structure)}")
    total_alimentadores = sum(len(alims) for alims in grid_structure.values())
    total_transformadores = sum(len(trafos) for alims in grid_structure.values() for trafos in alims.values())
    print(f"Total alimentadores: {total_alimentadores}")
    print(f"Total transformadores: {total_transformadores}")
    
    return grid_structure

if __name__ == "__main__":
    topology = analyze_grid_topology()