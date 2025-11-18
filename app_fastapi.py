import uvicorn
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware


# --- CONFIGURACIÓN CLAVE ---
# Asegúrate de que estos nombres coincidan con tus archivos
MODELO_PATH = "modelo/modelo_svd.pkl"
METADATOS_PATH = "combineds_ratings.csv" # cambiado estabaa dataset combineted_ratings.csv
# ---------------------------

# Inicializar FastAPI
app = FastAPI(title="API de Recomendación de Películas para Flutter")

# Agregar middleware CORS justo después de crear app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Aquí defines qué orígenes permites. "*" para permitir todos
    allow_credentials=True,
    allow_methods=["*"],  # Métodos HTTP permitidos
    allow_headers=["*"],  # Headers permitidos
)

# Variables globales para el modelo, metadatos y la lista de películas vistas
model = None
movie_data = None
all_movie_ids = []

# Definición del esquema de la respuesta (para Flutter)
class Recomendacion(BaseModel):
    """Esquema de datos que se enviará a Flutter."""
    movieId: int
    title: str
    poster_url: str
    estimated_rating: float

# --- FUNCIONES DE CARGA ---

def load_model_and_data():
    """Carga el modelo SVD y los metadatos (combined_ratings.csv) al iniciar la API."""
    global model, movie_data, all_movie_ids, user_ratings_df

    # 1. Cargar el Modelo SVD
    try:
        with open(MODELO_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Modelo SVD cargado desde: {MODELO_PATH}")
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo del modelo en {MODELO_PATH}.")
        raise RuntimeError("Modelo no encontrado.")

    # 2. Cargar los Metadatos y Ratings (desde combined_ratings.csv)
    try:
        df_combined = pd.read_csv(METADATOS_PATH)
        
        # Eliminar las filas duplicadas de ratings para obtener una única entrada de metadatos por movieId.
        movie_data = df_combined[['movieId', 'title', 'poster_url']].drop_duplicates(subset=['movieId']).set_index('movieId')
        
        # Lista de todos los MovieId únicos para iterar
        all_movie_ids = movie_data.index.unique().tolist()

        # Dataframe solo con ratings para filtrar películas ya vistas
        user_ratings_df = df_combined[['userId', 'movieId']].copy()

        print(f"✅ Metadatos de películas cargados: {len(movie_data)} películas únicas.")
        print(f"✅ Tabla de interacciones cargada para exclusión.")
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo de metadatos en {METADATOS_PATH}")
        raise RuntimeError("Metadatos de películas no encontrados.")


# --- ENDPOINTS ---

@app.on_event("startup")
async def startup_event():
    """Evento que se ejecuta al iniciar la aplicación para cargar recursos críticos."""
    try:
        load_model_and_data()
    except RuntimeError as e:
        print(f"Falló la carga crítica de recursos: {e}")
        # En un entorno de producción, aquí se manejaría el cierre seguro del servidor.


@app.get("/")
def read_root():
    """Endpoint de prueba de salud."""
    return {"status": "ok", "message": "Recommender API is running."}


@app.get("/recommend/{user_id}", response_model=List[Recomendacion])
def get_recommendations(user_id: int, top_n: int = 10):
    
    if model is None or movie_data is None:
        raise HTTPException(status_code=503, detail="Modelo o metadatos no cargados.")

    # 1. Filtrar películas vistas del usuario
    movies_seen = user_ratings_df[user_ratings_df['userId'] == user_id]["movieId"].tolist()

    # 2. Validar existencia de usuario
    if len(movies_seen) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"El usuario {user_id} NO existe en el dataset."
        )

    # 3. Películas no vistas
    candidate_movies = [mid for mid in all_movie_ids if mid not in movies_seen]

    predictions = []

    for movie_id in candidate_movies:
        pred = model.predict(user_id, movie_id)   # <--- tipo INT correcto
        predictions.append((movie_id, pred.est))

    # 4. Ordenar y retornar
    predictions.sort(key=lambda x: x[1], reverse=True)

    recommendations = []

    for movie_id, estimated_rating in predictions[:top_n]:
        row = movie_data.loc[movie_id]
        recommendations.append(Recomendacion(
            movieId=movie_id,
            title=row['title'],
            poster_url=row['poster_url'],
            estimated_rating=round(estimated_rating, 3)
        ))

    return recommendations


# --- EJECUTAR LA APLICACIÓN ---
if __name__ == "__main__":
    # INSTRUCCIÓN CLAVE PARA INICIAR:
    print("\n=======================================================================")
    print("Para iniciar el servidor, abre tu terminal en este directorio y ejecuta:")
    print("uvicorn app_fastapi:app --reload --port 8000")
    print("Luego, prueba con un ID de usuario en tu navegador:")
    print("http://127.0.0.1:8000/recommend/77188569?top_n=5")
    print("=======================================================================")
    # Nota: uvicorn.run() se usa para despliegue, el comando de terminal es más común para desarrollo.