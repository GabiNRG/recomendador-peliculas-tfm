## VERSIONES: Python==3.12.11, chromadb==1.0.20, sentence-transformers==5.1.0

# ================================================
# Streamlit App - Recomendador de Películas con ChromaDB + Groq
# ================================================
import streamlit as st
import pandas as pd
import chromadb
from groq import Groq
import py7zr
import os
import gdown  # Para descargar desde Google Drive

# ================================================
# Configuración inicial
# ================================================
st.set_page_config(page_title="🎬 Recomendador de Películas", layout="wide")
st.title("🎬 Recomendador de Películas con LLM + ChromaDB")

# API Key de Groq
# GROQ_API_KEY = os.environ["GROQ_API_KEY"]
# API Key de Groq (segura desde Streamlit Secrets)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# Rutas y parámetros
DB_PATH = "data/peliculas_ebd"
COLLECTION_NAME = "peliculas"
MODEL_NAME = "hiiamsid/sentence_similarity_spanish_es"
LLM_MODEL = "llama-3.1-8b-instant"
DATA_ARCHIVE = "data/peliculas.7z"

# Google Drive ID del archivo .7z
GOOGLE_DRIVE_FILE_ID = "https://drive.google.com/file/d/1tKgUHexiw2hnPJsWyL_wdeujAqgBm5-p/view?usp=sharing"

# ================================================
# Descargar y descomprimir dataset si no existe
# ================================================
if not os.path.exists("data"):
    st.info("📦 Descargando dataset de Google Drive, espera un momento...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", DATA_ARCHIVE, quiet=False)
    st.info("📦 Descomprimiendo dataset...")
    with py7zr.SevenZipFile(DATA_ARCHIVE, mode='r') as archive:
        archive.extractall(path="data")
    st.success("✅ Dataset descargado y descomprimido correctamente.")


# ================================================
# Cargar colección
# ================================================
@st.cache_resource
def cargar_coleccion(model_name=MODEL_NAME, db_path=DB_PATH, collection_name=COLLECTION_NAME):
    # client_chroma = chromadb.PersistentClient(path=db_path) # Versión anterior sin Settings y en local
    from chromadb.config import Settings
    client_chroma = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(
            chroma_db_impl="duckdb+parquet", # Para que funcione en Streamlit Cloud
            persist_directory=db_path
        )
    )
    from chromadb.utils import embedding_functions
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    collection = client_chroma.get_collection(name=collection_name, embedding_function=emb_fn)

    return collection

# ================================================
# Consulta a la colección
# ================================================
def buscar_peliculas(collection, query, n_results=3):
    all_data = collection.get()
    textos = [m["title"] for m in all_data["metadatas"]]
    textos_en = [m["title_orig"] for m in all_data["metadatas"]]

    if ((query in textos) | (query in textos_en)):
        if query in textos:
            idx = textos.index(query) # Si la query coincide con algun titulo de una pelicula, entonces se busca elementos cercanos a dicho elemento:
        else:
            idx = textos_en.index(query)

        descripcion_existente = all_data["documents"][idx]

        results = collection.query(
            query_texts=[descripcion_existente],  # aquí usamos el texto de la pelicula, emb_fn lo vectoriza
            n_results=n_results
        )

    else:
        results = collection.query(query_texts=[query], n_results=n_results)
    
    for idx, meta in enumerate(results["metadatas"][0]):
        recomendaciones = (query, results['documents'][0])
        
    return recomendaciones, results

# ================================================
# Consulta al LLM de Groq
# ================================================
@st.cache_resource
def cargar_cliente_llm(api_key):
    return Groq(api_key=api_key)

def consultar_llm(client, query, recomendaciones):
    recomendacion_texto = ".\n".join(recomendaciones[1])
    prompt = f"""
Consulta de usuario al modelo recomendador de películas:
{query}

---
En el texto anterior aparece una consulta a un modelo recomendador de películas, aparecerá aparecerá el nombre de una película o un género de una película, quiero que selecciones razonadamente las películas con las que esta relacionada a partir de los siguientes datos de contexto.
En caso de hacerlo, me gustaría que también referenciases los motivos por los que has utilizado esa información y no otra, así como la información que has utilizado.
Si la información proporcionada no es de utilidad, prefiero que me digas que no tienes información suficiente para responder a la pregunta.
Muestra el nombre de la película recomendada que sea diferente a la anterior película.
Finalmente, muestra una respuesta corta y piensa en profundidad en la respuesta. Céntrate exclusivamente en las películas recomendadas que aparecen a continuación:
---

Películas recomendadas por el modelo recomendador de películas:
{recomendacion_texto}
"""

    response = client.chat.completions.create(
        model= LLM_MODEL, #llama-3.1-8b-instant
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente cinéfilo experto en películas de todos los géneros, que habla en castellano. "
            "Cuando un usuario te da uno o más títulos de películas o géneros, recomiendas esas películas similares, incluidas en el contexto, en base a la consulta de usuario al modelo recomendador de películas"
            "explicando por qué podrían gustarle. Intenta no recomendar películas que sean la misma por la que se está preguntando en la consulta anterior traduciendo todos los títulos de las películas a castellano. Usa un lenguaje en castellano, claro, amigable y útil pero con respuestas cortas.")
            },
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# ================================================
# UI en Streamlit
# ================================================
st.sidebar.header("⚙️ Configuración")
num_resultados = st.sidebar.slider("Número de recomendaciones máximas por query", 1, 10, 3)

# Checkbox para mostrar u ocultar resultados
mostrar_resultados = st.sidebar.checkbox("Mostrar resultados del sistema de embeddings", value=True)

# Lista de modelos gratuitos permitidos
AVAILABLE_MODELS = [ # Ordenados de mayor a menor tamaño
    "openai/gpt-oss-120b",          # 120B parámetros (el más grande de la lista)
    "llama-3.3-70b-versatile",      # 70B parámetros, versátil
    "deepseek-r1-distill-llama-70b",# 70B distilled, sigue siendo muy potente
    "llama3-70b-8192",              # 70B parámetros
    "qwen/qwen3-32b",               # 32B parámetros      # en desuso: "mixtral-8x7b-32768",
    "openai/gpt-oss-20b",           # 20B parámetros
    "gemma2-9b-it",                 # 9B parámetros
    "llama3-8b-8192",               # 8B parámetros
    "llama-3.1-8b-instant",         # 8B optimizado para velocidad (menos preciso que el normal)
    "compound-beta-mini"            # Modelo pequeño experimental
]
# Selección de modelo único con valor por defecto
selected_model = st.sidebar.radio(
    "Selecciona el modelo LLM que quieres utilizar (estan ordenados de mayor a menor tamaño):",
    AVAILABLE_MODELS,
    index=AVAILABLE_MODELS.index("llama-3.1-8b-instant")  # Por defecto este
)
# Guardar el modelo en variable global (o sesión)
LLM_MODEL = selected_model
st.sidebar.write(f"✅ Modelo en uso: **{LLM_MODEL}**")


# Inicializar recursos cacheados
collection = cargar_coleccion()
llm_client = cargar_cliente_llm(GROQ_API_KEY)

# Caja de texto tipo chat
query = st.text_input("Escribe el título de una película o un género para obtener recomendaciones, si quieres buscar por una película en concreto, escribe unicamente el título de la película:",)

if query:
    with st.spinner("🔎 Buscando películas relacionadas..."):
        # Buscar en Chroma
        recomendaciones, results = buscar_peliculas(collection, query, n_results=num_resultados)

        # Mostrar resultados de Chroma si se marca el checkbox
        if mostrar_resultados: # st.checkbox("Mostrar resultados de ChromaDB", value=True):
            if recomendaciones:
                st.subheader("📊 Películas encontradas en la base de datos")
                st.write(f"**Recomendaciones encontradas en la colección de películas para la consulta anterior:** {len(recomendaciones[1])} películas")
                for idx, meta in enumerate(results["metadatas"][0]):
                    st.write(f"**{idx+1}. {meta['title']} ({meta['release_date']})** | ⭐ {meta['vote_average']}")
                    st.caption(results["documents"][0][idx])
            else:
                st.write("No se encontraron recomendaciones en la colección de películas para la consulta anterior.")
                
        

        # Consultar al LLM con Groq
        st.subheader("🤖 Recomendación del asistente")

        respuesta_final = consultar_llm(llm_client, query, recomendaciones)
        st.success(respuesta_final)
