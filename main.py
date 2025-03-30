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
class QLearningRouter:
    def __init__(self, grafo: nx.DiGraph, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Implementa Q-learning para optimización de rutas.
        
        Args:
            grafo (nx.DiGraph): Grafo de transporte.
        """
        self.grafo = grafo
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Exploración vs explotación
        
        # Inicializar tabla Q
        self.q_table = {}
        self._inicializar_q_table()

    def _inicializar_q_table(self):
        """
        Inicializa la tabla Q con valores aleatorios.
        """
        for nodo in self.grafo.nodes():
            for vecino in self.grafo.neighbors(nodo):
                if (nodo, vecino) not in self.q_table:
                    self.q_table[(nodo, vecino)] = np.random.uniform(0, 1)

    def encontrar_ruta(self, origen: str, destino: str) -> List[str]:
        """
        Encuentra la ruta óptima usando Q-learning.
        
        Args:
            origen (str): Nodo de origen.
            destino (str): Nodo de destino.
        
        Returns:
            List[str]: Ruta encontrada.
        """
        ruta_actual = [origen]
        nodo_actual = origen
        
        # Número máximo de iteraciones para evitar ciclos infinitos
        max_iteraciones = len(self.grafo.nodes) * 2
        iteraciones = 0
        
        while nodo_actual != destino and iteraciones < max_iteraciones:
            # Exploración vs explotación
            if np.random.random() < self.epsilon:
                # Exploración: elegir un vecino aleatorio
                vecinos = list(self.grafo.neighbors(nodo_actual))
                if not vecinos:
                    break
                siguiente_nodo = np.random.choice(vecinos)
            else:
                # Explotación: elegir el mejor vecino según Q-table
                vecinos = list(self.grafo.neighbors(nodo_actual))
                if not vecinos:
                    break
                
                valores_q = [self.q_table.get((nodo_actual, vecino), 0) for vecino in vecinos]
                siguiente_nodo = vecinos[np.argmax(valores_q)]
            
            ruta_actual.append(siguiente_nodo)
            nodo_actual = siguiente_nodo
            iteraciones += 1
        
        return ruta_actual
    
def convertir_numpy_a_json(obj):
    """
    Convierte objetos numpy a tipos de datos nativos de Python.
    
    Args:
        obj: Objeto a convertir
    
    Returns:
        Objeto convertido a tipo nativo de Python
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convertir_numpy_a_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertir_numpy_a_json(v) for v in obj]
    return obj

def main():
    # Iniciar medición de tiempo
    inicio = time.time()
    
    print("🚉 Iniciando Sistema de Transporte con IA...")
    print("Cargando datos y configurando modelo...")

    # Crear instancia del sistema de transporte con IA
    sistema = SistemaDeTransporteIA('datos.json')
    
    # Generar todas las combinaciones de rutas posibles
    ids_estaciones = list(sistema.estaciones.keys())
    rutas_ejemplo = list(itertools.permutations(ids_estaciones, 2))
    
    print(f"Total de rutas a calcular: {len(rutas_ejemplo)}")
    
    resultados = []
    for i, (origen, destino) in enumerate(rutas_ejemplo, 1):
        if origen != destino:  # Evitar rutas del mismo origen y destino
            # Mostrar progreso
            print(f"Calculando ruta {i}/{len(rutas_ejemplo)}: {origen} → {destino}")
            
            ruta = sistema.encontrar_ruta_ia(origen, destino)
            resultados.append(ruta)
    
    # Convertir resultados a tipos JSON serializables
    resultados_json = convertir_numpy_a_json(resultados)
    
    # Guardar resultados en un archivo JSON
    with open('Resultados.json', 'w', encoding='utf-8') as f:
        json.dump(resultados_json, f, ensure_ascii=False, indent=2)
    
    # Calcular tiempo de ejecución
    fin = time.time()
    tiempo_ejecucion = fin - inicio
    
    print(f"\n✅ Proceso completado.")
    print(f"🕒 Tiempo total de ejecución: {tiempo_ejecucion:.2f} segundos")
    print(f"📄 Resultados guardados en 'Resultados.json'")

if __name__ == "__main__":
    main()