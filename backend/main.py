from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
import numpy as np
from typing import List, Dict, Optional

app = FastAPI(title="Grid Analytics API", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de datos
class GridNode(BaseModel):
    id: str
    name: str
    type: str  # subestacion, alimentador, transformador, circuito
    coordx: Optional[float] = None
    coordy: Optional[float] = None
    potencia: Optional[float] = None
    usuarios: Optional[int] = None
    parent: Optional[str] = None
    children: List[str] = []

class GridStats(BaseModel):
    total_subestaciones: int
    total_alimentadores: int
    total_transformadores: int
    total_circuitos: int
    potencia_total: float
    usuarios_total: int
    area_geografica: Dict[str, float]

class PainPoint(BaseModel):
    id: str
    tipo: str
    descripcion: str
    severidad: str  # alta, media, baja
    coordx: float
    coordy: float
    potencia_afectada: float
    usuarios_afectados: int
    recomendacion: str

# Cargar datos al iniciar
try:
    df = pd.read_csv("../public/Mediciones Originales CEB .csv")
    with open("../grid_topology.json", "r", encoding="utf-8") as f:
        grid_topology = json.load(f)
except FileNotFoundError:
    df = None
    grid_topology = {}

@app.get("/")
async def root():
    return {"message": "Grid Analytics API - CEB Network Analysis"}

@app.get("/api/grid/stats", response_model=GridStats)
async def get_grid_stats():
    """Obtiene estadísticas generales de la red"""
    if df is None:
        raise HTTPException(status_code=500, detail="Datos no disponibles")
    
    coords_validas = df.dropna(subset=['Coordx', 'Coordy'])
    
    stats = GridStats(
        total_subestaciones=len(df['Historial Ago 21- May 25.Subestacion'].dropna().unique()),
        total_alimentadores=len(df['Idalimentador'].dropna().unique()),
        total_transformadores=len(df['Idtransformador'].dropna().unique()),
        total_circuitos=len(df['Codigoct'].dropna().unique()),
        potencia_total=float(df['POTENCIA'].sum()),
        usuarios_total=int(df['Usuarios Transformador'].sum()),
        area_geografica={
            "min_x": float(coords_validas['Coordx'].min()),
            "max_x": float(coords_validas['Coordx'].max()),
            "min_y": float(coords_validas['Coordy'].min()),
            "max_y": float(coords_validas['Coordy'].max()),
            "center_x": float(coords_validas['Coordx'].mean()),
            "center_y": float(coords_validas['Coordy'].mean())
        }
    )
    
    return stats

@app.get("/api/grid/topology")
async def get_grid_topology():
    """Obtiene la topología completa de la red"""
    return grid_topology

@app.get("/api/grid/nodes")
async def get_grid_nodes():
    """Obtiene todos los nodos de la red en formato plano"""
    if df is None:
        raise HTTPException(status_code=500, detail="Datos no disponibles")
    
    nodes = []
    
    # Procesar cada registro
    for _, row in df.iterrows():
        subestacion = str(row.get('Historial Ago 21- May 25.Subestacion', 'DESCONOCIDA'))
        alimentador = str(row.get('Idalimentador', 'DESCONOCIDO'))
        transformador = str(row.get('Idtransformador', 'DESCONOCIDO'))
        circuito = str(row.get('Codigoct', 'DESCONOCIDO'))
        
        # Crear nodo para el transformador/circuito
        node = GridNode(
            id=f"{subestacion}_{alimentador}_{transformador}_{circuito}",
            name=f"CT-{circuito}",
            type="transformador",
            coordx=float(row['Coordx']) if pd.notna(row['Coordx']) else None,
            coordy=float(row['Coordy']) if pd.notna(row['Coordy']) else None,
            potencia=float(row['POTENCIA']) if pd.notna(row['POTENCIA']) else None,
            usuarios=int(row['Usuarios Transformador']) if pd.notna(row['Usuarios Transformador']) else None,
            parent=f"{subestacion}_{alimentador}",
            children=[]
        )
        nodes.append(node.dict())
    
    return nodes

@app.get("/api/grid/pain-points", response_model=List[PainPoint])
async def get_pain_points():
    """Identifica puntos de dolor en la red"""
    if df is None:
        raise HTTPException(status_code=500, detail="Datos no disponibles")
    
    pain_points = []
    
    # Análisis de sobrecarga por densidad de usuarios
    df_coords = df.dropna(subset=['Coordx', 'Coordy', 'Usuarios Transformador', 'POTENCIA'])
    
    for _, row in df_coords.iterrows():
        usuarios = row['Usuarios Transformador']
        potencia = row['POTENCIA']
        
        # Identificar sobrecarga por ratio usuarios/potencia
        if usuarios > 0 and potencia > 0:
            ratio = usuarios / potencia
            
            if ratio > 1.0:  # Más de 1 usuario por kVA indica posible sobrecarga
                severidad = "alta" if ratio > 1.5 else "media"
                
                pain_point = PainPoint(
                    id=f"overload_{row['Codigoct']}",
                    tipo="sobrecarga",
                    descripcion=f"Alto ratio usuarios/potencia: {ratio:.2f} usuarios/kVA",
                    severidad=severidad,
                    coordx=float(row['Coordx']),
                    coordy=float(row['Coordy']),
                    potencia_afectada=float(potencia),
                    usuarios_afectados=int(usuarios),
                    recomendacion="Considerar instalación de parque solar para reducir demanda de red"
                )
                pain_points.append(pain_point)
    
    # Análisis de baja potencia instalada
    potencia_media = df_coords['POTENCIA'].mean()
    df_baja_potencia = df_coords[df_coords['POTENCIA'] < potencia_media * 0.5]
    
    for _, row in df_baja_potencia.iterrows():
        if row['Usuarios Transformador'] > 50:  # Muchos usuarios pero poca potencia
            pain_point = PainPoint(
                id=f"lowpower_{row['Codigoct']}",
                tipo="baja_potencia",
                descripcion=f"Baja potencia instalada ({row['POTENCIA']}kVA) para {row['Usuarios Transformador']} usuarios",
                severidad="media",
                coordx=float(row['Coordx']),
                coordy=float(row['Coordy']),
                potencia_afectada=float(row['POTENCIA']),
                usuarios_afectados=int(row['Usuarios Transformador']),
                recomendacion="Incrementar capacidad instalada o instalar generación distribuida"
            )
            pain_points.append(pain_point)
    
    return pain_points[:50]  # Limitar a 50 puntos más críticos

@app.get("/api/grid/subestaciones")
async def get_subestaciones():
    """Obtiene información de todas las subestaciones"""
    if df is None:
        raise HTTPException(status_code=500, detail="Datos no disponibles")
    
    subestaciones = []
    df_grouped = df.groupby('Historial Ago 21- May 25.Subestacion').agg({
        'POTENCIA': 'sum',
        'Usuarios Transformador': 'sum',
        'Coordx': 'mean',
        'Coordy': 'mean',
        'Idalimentador': 'nunique'
    }).reset_index()
    
    for _, row in df_grouped.iterrows():
        subestacion = {
            "id": str(row['Historial Ago 21- May 25.Subestacion']),
            "name": f"Subestación {row['Historial Ago 21- May 25.Subestacion']}",
            "potencia_total": float(row['POTENCIA']),
            "usuarios_total": int(row['Usuarios Transformador']),
            "alimentadores": int(row['Idalimentador']),
            "coordx": float(row['Coordx']) if pd.notna(row['Coordx']) else None,
            "coordy": float(row['Coordy']) if pd.notna(row['Coordy']) else None
        }
        subestaciones.append(subestacion)
    
    return subestaciones

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)