# Sistema de Optimización de Rutas de Transporte con IA

## Descripción General
Este proyecto implementa un sistema de optimización de rutas de transporte utilizando técnicas avanzadas de IA, incluyendo Q-learning y redes neuronales, para encontrar rutas óptimas entre estaciones.

## Requisitos Previos

### Versión de Python
- **Versión Recomendada**: Python 3.8 - 3.11
- **Versión Mínima**: Python 3.7+

### Bibliotecas Necesarias
Antes de ejecutar el proyecto, asegúrate de tener instaladas las siguientes bibliotecas de Python:
- numpy
- networkx
- scikit-learn
- tensorflow
- json (generalmente incluido por defecto)

### Instalación
Puedes instalar las bibliotecas necesarias usando pip:

```bash
pip install numpy networkx scikit-learn tensorflow
```

## Estructura del Proyecto
- `main.py`: Script principal que contiene las clases del sistema de transporte con IA
- `datos.json`: Archivo de entrada con información de estaciones y conexiones
- `Resultados.json`: Archivo de salida con resultados de optimización de rutas

## Pasos de Ejecución

1. Preparar Datos de Entrada
   Crea un archivo `datos.json` con la siguiente estructura:
   ```json
   {
     "estaciones": [
       {
         "id": "EST1",
         "lineas": ["L1", "L2"],
         "nombre": "Estación Central"
       }
     ],
     "conexiones": [
       {
         "origen": "EST1",
         "destino": "EST2",
         "tiempo": 10,
         "distancia": 5,
         "linea": "L1"
       }
     ],
     "reglas": [
       {
         "tipo": "mantenimiento_tramo",
         "origen": "EST1",
         "destino": "EST2"
       }
     ]
   }
   ```

2. Ejecutar el Script
   ```bash
   python main.py
   ```

## Características Principales
- Optimización de rutas mediante Q-learning
- Predicción de tiempos de viaje con redes neuronales
- Consideración de reglas de mantenimiento y congestión
- Análisis detallado de rutas

## Salida
Después de la ejecución, el script genera un archivo `Resultados.json` que contiene:
- Rutas óptimas entre todas las combinaciones de estaciones
- Tiempos de viaje predichos
- Información detallada de las estaciones

## Personalización
Modifica `datos.json` para:
- Agregar/eliminar estaciones
- Definir reglas de conexión
- Ajustar parámetros de mantenimiento y congestión
