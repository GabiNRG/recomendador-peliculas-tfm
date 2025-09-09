## VERSIONES: Python==3.12.11, chromadb==1.0.20, sentence-transformers==5.1.0

# ================================================
# Streamlit App - Recomendador de Películas con ChromaDB + Groq
# ================================================
import streamlit as st
import pickle
import gzip
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq

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
DATA_PATH = "data/peliculas_data.pkl.gz"
MODEL_NAME = "hiiamsid/sentence_similarity_spanish_es"
LLM_MODEL = "llama-3.1-8b-instant"


# ================================================
# Cargar colección desde .pkl con embeddings precomputados
# ================================================
@st.cache_resource
def cargar_coleccion(path=DATA_PATH):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            collection_data = pickle.load(f)
    else:
        with open(path, "rb") as f:
            collection_data = pickle.load(f)
    return collection_data


# ================================================
# Cargar modelo SentenceTransformer
# ================================================
@st.cache_resource
def cargar_modelo(model_name=MODEL_NAME):
    return SentenceTransformer(model_name)


# ================================================
# Consulta a la colección
# ================================================
def buscar_peliculas(collection_data, query, model_name=MODEL_NAME, n_results=3):
    model = cargar_modelo(model_name)
    # Extraer textos
    textos = [m["title"] for m in collection_data["metadatas"]]
    textos_en = [m["title_orig"] for m in collection_data["metadatas"]]
    
    # Si la query coincide exactamente con un título
    if query in textos:
        idx = textos.index(query)
        query_emb = collection_data["embeddings"][idx]  # usar embedding de la propia película
    elif query in textos_en:
        idx = textos_en.index(query)
        query_emb = collection_data["embeddings"][idx]
    else:
        # Si no coincide, generar embedding con el modelo
        query_emb = model.encode([query])[0]

    # Cálculo de similitud coseno
    embeddings = np.array(collection_data["embeddings"])
    similarities = embeddings @ query_emb / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb))
    idxs = np.argsort(-similarities)[:n_results]

    docs = [collection_data["documents"][i] for i in idxs]
    metas = [collection_data["metadatas"][i] for i in idxs]

    return (query, docs), {"documents": [docs], "metadatas": [metas]}


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

