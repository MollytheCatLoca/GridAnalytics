'use client'

import { MapContainer, TileLayer, Marker, Popup, CircleMarker } from 'react-leaflet'
import { Icon } from 'leaflet'
import 'leaflet/dist/leaflet.css'

interface PainPoint {
  id: string
  tipo: string
  descripcion: string
  severidad: string
  coordx: number
  coordy: number
  potencia_afectada: number
  usuarios_afectados: number
  recomendacion: string
}

interface GridMapProps {
  center: [number, number]
  painPoints: PainPoint[]
}

// Fix leaflet icon issue in Next.js
const createIcon = (color: string) => new Icon({
  iconUrl: `data:image/svg+xml;base64,${btoa(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 25 41" width="25" height="41">
      <path fill="${color}" stroke="#fff" stroke-width="1" d="M12.5 0C5.6 0 0 5.6 0 12.5c0 7.8 12.5 28.5 12.5 28.5S25 20.3 25 12.5C25 5.6 19.4 0 12.5 0z"/>
      <circle fill="#fff" cx="12.5" cy="12.5" r="4"/>
    </svg>
  `)}`,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34]
})

const getSeverityColor = (severidad: string) => {
  switch (severidad) {
    case 'alta': return '#dc2626'
    case 'media': return '#ea580c'
    case 'baja': return '#16a34a'
    default: return '#6b7280'
  }
}

export default function GridMap({ center, painPoints }: GridMapProps) {
  return (
    <MapContainer
      center={center}
      zoom={11}
      style={{ height: '500px', width: '100%' }}
      className="rounded-lg"
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      
      {painPoints.map((point) => (
        <CircleMarker
          key={point.id}
          center={[point.coordy, point.coordx]}
          radius={8}
          pathOptions={{
            color: getSeverityColor(point.severidad),
            fillColor: getSeverityColor(point.severidad),
            fillOpacity: 0.7,
            weight: 2
          }}
        >
          <Popup>
            <div className="p-2">
              <h3 className="font-semibold text-sm capitalize">
                {point.tipo.replace('_', ' ')}
              </h3>
              <p className="text-xs text-gray-600 mt-1">
                {point.descripcion}
              </p>
              <div className="mt-2 text-xs">
                <div>
                  <strong>Severidad:</strong> 
                  <span className={`ml-1 px-1 py-0.5 rounded text-white ${
                    point.severidad === 'alta' ? 'bg-red-600' :
                    point.severidad === 'media' ? 'bg-orange-600' : 'bg-green-600'
                  }`}>
                    {point.severidad.toUpperCase()}
                  </span>
                </div>
                <div className="mt-1">
                  <strong>Usuarios:</strong> {point.usuarios_afectados}
                </div>
                <div>
                  <strong>Potencia:</strong> {point.potencia_afectada.toFixed(1)} kVA
                </div>
                <div className="mt-2 text-blue-600 font-medium">
                  {point.recomendacion}
                </div>
              </div>
            </div>
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  )
}