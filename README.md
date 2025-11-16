# movie-recommender-api
API REST incorporada a traves de FastAPI con Python + entremaniento con filtrado colaborativo con datasets de MovieLens

🎬 Sistema de Recomendación con MovieLens usando Filtrado Colaborativo (SVD)

Este proyecto implementa un Sistema de Recomendación de Películas basado en el dataset MovieLens, utilizando modelos de Filtrado Colaborativo mediante SVD (Singular Value Decomposition) entrenado con la librería scikit-surprise.

El objetivo es construir un motor inteligente capaz de predecir la calificación que un usuario le daría a una película aún no vista, y generar recomendaciones personalizadas.

🚀 Impacto y utilidad del modelo

El filtrado colaborativo es uno de los métodos más utilizados en empresas como:

Netflix

Amazon

YouTube

Spotify

Porque aprende patrones de comportamiento entre usuarios y productos.

¿Qué beneficios aporta?

🎯 Predicciones precisas basadas en gustos reales de los usuarios

🤖 Aprendizaje automático continuo mientras se agregan nuevos datos

🎞️ Recomendaciones personalizadas que aumentan la interacción

⚙️ Modelo ligero y eficiente, perfecto para APIs en producción

📈 Generalización ante nuevos usuarios o nuevas películas

🧠 ¿Por qué MovieLens?

MovieLens es el dataset estándar para evaluar sistemas de recomendación.
Ofrece:

Más de 100,000 o 1M de calificaciones (según versión)

Datos limpios y confiables

Amplia comunidad científica usándolo

Facilidad para reproducir resultados y comparar modelos

Es ideal para proyectos académicos, prototipos y sistemas reales.

🏗️ Arquitectura del Proyecto

FastAPI → Servidor backend

Modelo entrenado SVD (model_svd.pkl)

Dataset MovieLens preprocesado

Endpoints REST para obtener predicciones y recomendaciones

Despliegue en Railway para acceso público

🔮 Futuras mejoras

Integrar contenido adicional (géneros, directores, tags)

Híbrido: colaborativo + basado en contenido

Motor real-time con actualización incremental

Dashboard de métricas y rendimiento
