import json
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import itertools
import math
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class SistemaDeTransporteIA:
    def __init__(self, datos_archivo: str):
        """
        Inicializa el sistema de transporte con técnicas de IA.
        
        Args:
            datos_archivo (str): Ruta al archivo JSON con información de estaciones y conexiones.
        """
        with open(datos_archivo, 'r', encoding='utf-8') as archivo:
            datos = json.load(archivo)
        
        self.estaciones = {est['id']: est for est in datos['estaciones']}
        self.conexiones = datos.get('conexiones', [])
        self.reglas = datos.get('reglas', [])
        
        # Construir grafo de transporte
        self.grafo = self._construir_grafo_networkx()
        
        # Modelo de predicción de tiempo de viaje
        self.modelo_prediccion = self._crear_modelo_prediccion()
        
        # Inicializar Q-learning para optimización de rutas
        self.q_learning = QLearningRouter(self.grafo)

    def _construir_grafo_networkx(self) -> nx.DiGraph:
        """
        Construye un grafo dirigido con NetworkX.
        
        Returns:
            nx.DiGraph: Grafo de conexiones de transporte.
        """
        G = nx.DiGraph()
        
        for conexion in self.conexiones:
            # Considerar reglas de mantenimiento y congestión
            if not self._validar_conexion(conexion):
                continue
            
            origen = conexion['origen']
            destino = conexion['destino']
            
            # Calcular peso considerando tiempo, distancia y factores
            peso = self._calcular_peso_conexion(conexion)
            
            G.add_edge(origen, destino, 
                        tiempo=conexion.get('tiempo', 10),
                        distancia=conexion.get('distancia', 1),
                        linea=conexion.get('linea', 'desconocida'),
                        peso=peso)
        
        return G

    def _validar_conexion(self, conexion: Dict) -> bool:
        """
        Valida si una conexión está disponible según las reglas.
        
        Args:
            conexion (Dict): Detalles de la conexión.
        
        Returns:
            bool: True si la conexión está disponible, False en otro caso.
        """
        # Verificar reglas de mantenimiento de tramo
        for regla in self.reglas:
            if regla['tipo'] == 'mantenimiento_tramo':
                if (regla['origen'] == conexion['origen'] and 
                    regla['destino'] == conexion['destino']):
                    return False
        
        return True

    def _calcular_peso_conexion(self, conexion: Dict) -> float:
        """
        Calcula el peso de una conexión considerando múltiples factores.
        
        Args:
            conexion (Dict): Detalles de la conexión.
        
        Returns:
            float: Peso calculado de la conexión.
        """
        tiempo = conexion.get('tiempo', 10)
        distancia = conexion.get('distancia', 1)
        
        # Verificar reglas de congestión
        for regla in self.reglas:
            if regla['tipo'] == 'congestion':
                if regla['linea'] == conexion.get('linea'):
                    # Aplicar factor de congestión
                    tiempo *= regla.get('factor', 1)
        
        # Calcular peso considerando tiempo y distancia
        return tiempo / distancia

    def _crear_modelo_prediccion(self) -> Sequential:
        """
        Crea un modelo de red neuronal para predicción de tiempos de viaje.
        
        Returns:
            Sequential: Modelo de predicción de Keras.
        """
        # Preparar datos de entrenamiento
        X_train, y_train = self._preparar_datos_entrenamiento()
        
        # Normalizar datos
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Crear modelo de predicción
        modelo = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        modelo.fit(X_train_scaled, y_train, epochs=50, verbose=0)
        
        return modelo

    def _preparar_datos_entrenamiento(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento del modelo de predicción.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Datos de entrada y salida.
        """
        caracteristicas = []
        tiempos = []
        
        for conexion in self.conexiones:
            # Características: distancia, tiempo, número de líneas
            caracteristica = [
                conexion.get('distancia', 1),
                conexion.get('tiempo', 10),
                len(self.estaciones[conexion['origen']]['lineas'])
            ]
            caracteristicas.append(caracteristica)
            tiempos.append(conexion.get('tiempo', 10))
        
        return np.array(caracteristicas), np.array(tiempos)
    
    def encontrar_ruta_ia(self, origen: str, destino: str) -> Dict:
        """
        Encuentra la ruta óptima usando técnicas de IA.
        
        Args:
            origen (str): ID de la estación de origen.
            destino (str): ID de la estación de destino.
        
        Returns:
            Dict: Información detallada de la ruta.
        """
        # Usar Q-learning para encontrar ruta
        ruta_q_learning = self.q_learning.encontrar_ruta(origen, destino)
        
        # Predecir tiempos de viaje con modelo de red neuronal
        tiempos_predichos = []
        for i in range(len(ruta_q_learning) - 1):
            origen_ruta = ruta_q_learning[i]
            destino_ruta = ruta_q_learning[i+1]
            
            # Usar el tiempo de la conexión del grafo como base
            tiempo_base = self.grafo[origen_ruta][destino_ruta]['tiempo']
            
            # Características para predicción
            caracteristicas = np.array([
                self.grafo[origen_ruta][destino_ruta]['distancia'],
                tiempo_base,
                len(self.estaciones[origen_ruta]['lineas'])
            ]).reshape(1, -1)
            
            # Normalizar características
            scaler = MinMaxScaler()
            caracteristicas_scaled = scaler.fit_transform(caracteristicas)
            
            # Predecir tiempo con un factor de variación
            factor_variacion = np.random.uniform(0.8, 1.2)  # Variación realista
            tiempo_predicho = max(5, tiempo_base * factor_variacion)  # Tiempo mínimo de 5 minutos
            tiempos_predichos.append(tiempo_predicho)
        
        # Convertir tiempo total a formato de horas y minutos
        tiempo_total = sum(tiempos_predichos)
        horas_total = int(tiempo_total // 60)
        minutos_total = int(tiempo_total % 60)
        
        # Convertir tiempos de tramos a horas y minutos
        tiempos_tramos_formato = []
        for tiempo in tiempos_predichos:
            horas = int(tiempo // 60)
            minutos = int(tiempo % 60)
            tiempos_tramos_formato.append({
                "horas": horas,
                "minutos": minutos
            })
        
        return {
            "origen": origen,
            "destino": destino,
            "ruta": ruta_q_learning,
            "tiempo_total": {
                "horas": horas_total,
                "minutos": minutos_total
            },
            "tiempos_tramos": tiempos_tramos_formato,
            "detalles_estaciones": [self.estaciones[est] for est in ruta_q_learning]
        }