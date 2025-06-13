#!/usr/bin/env python3
"""
Crear visualización topológica unifilar de la red CEB
Muestra troncales y ramales con jerarquía de red
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def create_network_topology():
    """Crea el grafo topológico de la red eléctrica"""
    print("=== CREANDO TOPOLOGÍA DE RED CEB ===\n")
    
    # Cargar datos
    df = pd.read_csv("../public/Mediciones Originales CEB .csv")
    print(f"✅ Datos cargados: {len(df)} registros")
    
    # Crear grafo de red
    G = nx.Graph()
    
    # Diccionarios para almacenar información
    subestaciones = {}
    alimentadores = {}
    transformadores = {}
    conexiones = []
    
    # Procesar datos y construir jerarquía
    print("\n📊 CONSTRUYENDO JERARQUÍA DE RED...")
    
    # 1. Identificar subestaciones únicas
    for _, row in df.dropna(subset=['Historial Ago 21- May 25.Subestacion', 'Coordx', 'Coordy']).iterrows():
        sub_id = str(row['Historial Ago 21- May 25.Subestacion'])
        if sub_id not in subestaciones:
            subestaciones[sub_id] = {
                'id': f"S_{sub_id}",
                'name': f"Subestación {sub_id}",
                'type': 'subestacion',
                'coordx': row['Coordx'],
                'coordy': row['Coordy'],
                'potencia': 0,
                'usuarios': 0
            }
    
    # 2. Procesar alimentadores y transformadores
    for _, row in df.iterrows():
        sub_id = str(row.get('Historial Ago 21- May 25.Subestacion', 'UNKNOWN'))
        alim_id = str(row.get('Idalimentador', 'UNKNOWN'))
        trafo_id = str(row.get('Idtransformador', 'UNKNOWN'))
        
        # Actualizar potencia y usuarios de subestación
        if sub_id in subestaciones and pd.notna(row['POTENCIA']):
            subestaciones[sub_id]['potencia'] += row['POTENCIA']
            subestaciones[sub_id]['usuarios'] += row.get('Usuarios Transformador', 0)
        
        # Crear nodo alimentador si no existe
        alim_key = f"{sub_id}_{alim_id}"
        if alim_id != 'UNKNOWN' and alim_key not in alimentadores:
            alimentadores[alim_key] = {
                'id': f"A_{alim_key}",
                'name': f"Alimentador {alim_id}",
                'type': 'alimentador',
                'subestacion': sub_id,
                'coordx': row.get('Coordx', 0),
                'coordy': row.get('Coordy', 0),
                'potencia': 0,
                'usuarios': 0
            }
        
        # Actualizar datos del alimentador
        if alim_key in alimentadores and pd.notna(row['POTENCIA']):
            alimentadores[alim_key]['potencia'] += row['POTENCIA']
            alimentadores[alim_key]['usuarios'] += row.get('Usuarios Transformador', 0)
        
        # Crear conexión subestación -> alimentador
        if sub_id in subestaciones and alim_key in alimentadores:
            conexiones.append((f"S_{sub_id}", f"A_{alim_key}"))
        
        # Crear nodo transformador
        trafo_key = f"{alim_key}_{trafo_id}_{row.get('Codigoct', '')}"
        if trafo_id != 'UNKNOWN' and pd.notna(row['Coordx']) and pd.notna(row['Coordy']):
            transformadores[trafo_key] = {
                'id': f"T_{trafo_key}",
                'name': f"CT-{row.get('Codigoct', trafo_id)}",
                'type': 'transformador',
                'alimentador': alim_key,
                'coordx': row['Coordx'],
                'coordy': row['Coordy'],
                'potencia': row.get('POTENCIA', 0),
                'usuarios': row.get('Usuarios Transformador', 0),
                'direccion': row.get('Direccion', '')
            }
            
            # Crear conexión alimentador -> transformador
            if alim_key in alimentadores:
                conexiones.append((f"A_{alim_key}", f"T_{trafo_key}"))
    
    print(f"✅ Subestaciones: {len(subestaciones)}")
    print(f"✅ Alimentadores: {len(alimentadores)}")
    print(f"✅ Transformadores: {len(transformadores)}")
    print(f"✅ Conexiones: {len(conexiones)}")
    
    # Agregar nodos al grafo
    for sub in subestaciones.values():
        G.add_node(sub['id'], **sub)
    
    for alim in alimentadores.values():
        G.add_node(alim['id'], **alim)
    
    # Solo agregar transformadores con más usuarios para simplificar visualización
    trafo_threshold = 50  # Solo mostrar transformadores con más de 50 usuarios
    for trafo in transformadores.values():
        if trafo['usuarios'] > trafo_threshold:
            G.add_node(trafo['id'], **trafo)
    
    # Agregar conexiones
    for edge in conexiones:
        if edge[0] in G.nodes() and edge[1] in G.nodes():
            G.add_edge(edge[0], edge[1])
    
    print(f"\n📊 GRAFO FINAL:")
    print(f"   - Nodos: {G.number_of_nodes()}")
    print(f"   - Conexiones: {G.number_of_edges()}")
    
    # Crear visualización
    create_topology_visualization(G, subestaciones, alimentadores, transformadores)
    
    # Crear mapa geográfico
    create_geographic_map(G, df, subestaciones)
    
    return G, subestaciones, alimentadores, transformadores

def create_topology_visualization(G, subestaciones, alimentadores, transformadores):
    """Crea visualización topológica unifilar"""
    print("\n🎨 CREANDO DIAGRAMA UNIFILAR...")
    
    # Configurar figura grande
    plt.figure(figsize=(24, 16))
    
    # Usar layout jerárquico
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Ajustar posiciones para jerarquía visual
    # Subestaciones arriba, alimentadores en medio, transformadores abajo
    for node, data in G.nodes(data=True):
        if data['type'] == 'subestacion':
            pos[node] = (pos[node][0], 1.0)
        elif data['type'] == 'alimentador':
            pos[node] = (pos[node][0], 0.0)
        elif data['type'] == 'transformador':
            pos[node] = (pos[node][0], -1.0)
    
    # Dibujar conexiones
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, width=1)
    
    # Dibujar nodos por tipo
    # Subestaciones - grandes, rojas
    sub_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'subestacion']
    nx.draw_networkx_nodes(G, pos, nodelist=sub_nodes, 
                          node_color='red', node_size=1000, 
                          node_shape='s', label='Subestaciones')
    
    # Alimentadores - medianos, azules
    alim_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'alimentador']
    nx.draw_networkx_nodes(G, pos, nodelist=alim_nodes,
                          node_color='blue', node_size=400,
                          node_shape='o', label='Alimentadores')
    
    # Transformadores - pequeños, verdes
    trafo_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'transformador']
    nx.draw_networkx_nodes(G, pos, nodelist=trafo_nodes,
                          node_color='green', node_size=200,
                          node_shape='v', label='Transformadores')
    
    # Etiquetas solo para subestaciones
    sub_labels = {n: G.nodes[n]['name'].replace('Subestación ', 'S') 
                  for n in sub_nodes}
    nx.draw_networkx_labels(G, pos, sub_labels, font_size=8)
    
    plt.title("DIAGRAMA UNIFILAR - RED ELÉCTRICA CEB\nJerarquía: Subestaciones → Alimentadores → Transformadores", 
              fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('red_ceb_unifilar.png', dpi=300, bbox_inches='tight')
    print("✅ Diagrama unifilar guardado: red_ceb_unifilar.png")
    plt.close()

def create_geographic_map(G, df, subestaciones):
    """Crea mapa geográfico de la red"""
    print("\n🗺️ CREANDO MAPA GEOGRÁFICO...")
    
    # Figura para mapa geográfico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # MAPA 1: Vista general con todos los componentes
    ax1.set_title("MAPA GEOGRÁFICO - RED CEB COMPLETA", fontsize=14, fontweight='bold')
    
    # Plotear por tipo con diferentes colores y tamaños
    for node, data in G.nodes(data=True):
        if data['type'] == 'subestacion':
            ax1.scatter(data['coordx'], data['coordy'], 
                       s=300, c='red', marker='s', alpha=0.8, 
                       edgecolors='darkred', linewidth=2)
            ax1.text(data['coordx'], data['coordy']+0.01, 
                    data['name'].replace('Subestación ', 'S'), 
                    fontsize=8, ha='center')
        elif data['type'] == 'alimentador':
            ax1.scatter(data['coordx'], data['coordy'], 
                       s=100, c='blue', marker='o', alpha=0.6)
        elif data['type'] == 'transformador':
            ax1.scatter(data['coordx'], data['coordy'], 
                       s=30, c='green', marker='v', alpha=0.5)
    
    # Dibujar conexiones
    for edge in G.edges():
        node1 = G.nodes[edge[0]]
        node2 = G.nodes[edge[1]]
        ax1.plot([node1['coordx'], node2['coordx']], 
                [node1['coordy'], node2['coordy']], 
                'gray', alpha=0.3, linewidth=0.5)
    
    ax1.set_xlabel('Longitud')
    ax1.set_ylabel('Latitud')
    ax1.grid(True, alpha=0.3)
    
    # MAPA 2: Mapa de calor por densidad de usuarios
    ax2.set_title("DENSIDAD DE USUARIOS POR ZONA", fontsize=14, fontweight='bold')
    
    # Crear mapa de calor
    coords = df.dropna(subset=['Coordx', 'Coordy', 'Usuarios Transformador'])
    scatter = ax2.scatter(coords['Coordx'], coords['Coordy'], 
                         c=coords['Usuarios Transformador'], 
                         cmap='YlOrRd', s=50, alpha=0.6)
    
    # Resaltar puntos críticos (más de 200 usuarios)
    critical = coords[coords['Usuarios Transformador'] > 200]
    ax2.scatter(critical['Coordx'], critical['Coordy'], 
               s=200, facecolors='none', edgecolors='red', 
               linewidth=2, label='Puntos críticos (>200 usuarios)')
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Usuarios por transformador')
    ax2.set_xlabel('Longitud')
    ax2.set_ylabel('Latitud')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('red_ceb_geografico.png', dpi=300, bbox_inches='tight')
    print("✅ Mapa geográfico guardado: red_ceb_geografico.png")
    plt.close()
    
    # Crear mapa de troncales principales
    create_trunk_map(G, df, subestaciones)

def create_trunk_map(G, df, subestaciones):
    """Crea mapa de troncales principales"""
    print("\n🛤️ CREANDO MAPA DE TRONCALES...")
    
    plt.figure(figsize=(16, 12))
    
    # Identificar troncales principales (subestaciones con más alimentadores)
    sub_connections = defaultdict(int)
    for node, data in G.nodes(data=True):
        if data['type'] == 'alimentador':
            sub_connections[data['subestacion']] += 1
    
    # Top 10 subestaciones principales
    main_subs = sorted(sub_connections.items(), key=lambda x: x[1], reverse=True)[:10]
    main_sub_ids = [sub[0] for sub in main_subs]
    
    # Plotear subestaciones principales más grandes
    for node, data in G.nodes(data=True):
        if data['type'] == 'subestacion':
            sub_id = node.replace('S_', '')
            if sub_id in main_sub_ids:
                size = 500 + (sub_connections[sub_id] * 50)
                plt.scatter(data['coordx'], data['coordy'], 
                           s=size, c='darkred', marker='s', 
                           edgecolors='black', linewidth=2)
                plt.text(data['coordx'], data['coordy']+0.015, 
                        f"S{sub_id}\n({sub_connections[sub_id]} alim)", 
                        fontsize=10, ha='center', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            else:
                plt.scatter(data['coordx'], data['coordy'], 
                           s=200, c='red', marker='s', alpha=0.5)
    
    # Dibujar conexiones troncales (solo entre subestaciones principales)
    for edge in G.edges():
        node1_data = G.nodes[edge[0]]
        node2_data = G.nodes[edge[1]]
        
        if (node1_data['type'] == 'subestacion' and 
            node2_data['type'] == 'alimentador' and
            node1_data['id'].replace('S_', '') in main_sub_ids):
            
            # Línea más gruesa para troncales principales
            plt.plot([node1_data['coordx'], node2_data['coordx']], 
                    [node1_data['coordy'], node2_data['coordy']], 
                    'darkblue', alpha=0.8, linewidth=2)
    
    plt.title("MAPA DE TRONCALES PRINCIPALES - RED CEB\nSubestaciones con mayor cantidad de alimentadores", 
              fontsize=16, fontweight='bold')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.grid(True, alpha=0.3)
    
    # Añadir leyenda con información
    plt.text(0.02, 0.98, 
             f"Total subestaciones: {len(subestaciones)}\n" +
             f"Subestaciones principales: {len(main_sub_ids)}\n" +
             f"Mayor concentración: S{main_subs[0][0]} ({main_subs[0][1]} alimentadores)",
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('red_ceb_troncales.png', dpi=300, bbox_inches='tight')
    print("✅ Mapa de troncales guardado: red_ceb_troncales.png")
    plt.close()

if __name__ == "__main__":
    try:
        G, subestaciones, alimentadores, transformadores = create_network_topology()
        
        print("\n=== RESUMEN DE TOPOLOGÍA ===")
        print(f"✅ Análisis completado exitosamente")
        print(f"📊 Archivos generados:")
        print(f"   - red_ceb_unifilar.png: Diagrama unifilar jerárquico")
        print(f"   - red_ceb_geografico.png: Mapa geográfico con densidad")
        print(f"   - red_ceb_troncales.png: Mapa de troncales principales")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()