'use client'

import { useState, useEffect } from 'react'
import { Activity, Zap, Users, MapPin, AlertTriangle } from 'lucide-react'

interface GridStats {
  total_subestaciones: number
  total_alimentadores: number
  total_transformadores: number
  total_circuitos: number
  potencia_total: number
  usuarios_total: number
  area_geografica: {
    min_x: number
    max_x: number
    min_y: number
    max_y: number
    center_x: number
    center_y: number
  }
}

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

export default function Home() {
  const [stats, setStats] = useState<GridStats | null>(null)
  const [painPoints, setPainPoints] = useState<PainPoint[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const statsResponse = await fetch('http://localhost:8000/api/grid/stats')
        if (statsResponse.ok) {
          const statsData = await statsResponse.json()
          setStats(statsData)
        }

        const painResponse = await fetch('http://localhost:8000/api/grid/pain-points')
        if (painResponse.ok) {
          const painData = await painResponse.json()
          setPainPoints(painData)
        }
      } catch (error) {
        console.error('Error cargando datos:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Cargando análisis de red CEB...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Zap className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Grid Analytics</h1>
                <p className="text-sm text-gray-600">Análisis de Red CEB - Cooperativa Eléctrica de Bolívar</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <MapPin className="h-8 w-8 text-blue-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      Subestaciones
                    </dt>
                    <dd className="text-2xl font-semibold text-gray-900">
                      {stats.total_subestaciones}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <Activity className="h-8 w-8 text-green-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      Potencia Total
                    </dt>
                    <dd className="text-2xl font-semibold text-gray-900">
                      {(stats.potencia_total / 1000).toFixed(1)} MW
                    </dd>
                  </dl>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <Users className="h-8 w-8 text-purple-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      Usuarios Totales
                    </dt>
                    <dd className="text-2xl font-semibold text-gray-900">
                      {stats.usuarios_total.toLocaleString()}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <AlertTriangle className="h-8 w-8 text-red-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      Puntos de Dolor
                    </dt>
                    <dd className="text-2xl font-semibold text-gray-900">
                      {painPoints.length}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">
              Puntos de Dolor Identificados
            </h2>
            <p className="mt-1 text-sm text-gray-600">
              Áreas críticas que requieren atención para optimización con energía solar
            </p>
          </div>
          <div className="divide-y divide-gray-200">
            {painPoints.slice(0, 10).map((point) => (
              <div key={point.id} className="px-6 py-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                        point.severidad === 'alta' 
                          ? 'bg-red-100 text-red-800'
                          : point.severidad === 'media'
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-green-100 text-green-800'
                      }`}>
                        {point.severidad.toUpperCase()}
                      </span>
                      <span className="text-sm font-medium text-gray-900 capitalize">
                        {point.tipo.replace('_', ' ')}
                      </span>
                    </div>
                    <p className="mt-1 text-sm text-gray-600">{point.descripcion}</p>
                    <p className="mt-2 text-xs text-blue-600 font-medium">
                      💡 {point.recomendacion}
                    </p>
                  </div>
                  <div className="text-right text-sm text-gray-500">
                    <div>{point.usuarios_afectados} usuarios</div>
                    <div>{point.potencia_afectada.toFixed(1)} kVA</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}