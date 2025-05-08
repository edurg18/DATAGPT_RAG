# Libreria de python necesarias para el RAG
#import langchain 
#import faiss
#import openai
#import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import utils, response_properties
import Config as config


file = config.FILE

# PASO 1 : Leer el archivo
path_pdf = f'''./{file}'''

texto = utils.read_pdf(path_pdf)


# PASO 2 : Dividimos el texto en Chunks
divided_text = utils.create_chunks(texto, 500)

# PASO 3 : Sacamos los embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

embeddings = model.encode(divided_text, convert_to_tensor=True)

# PASO 4 : Creamos un índice

# Convertir los embeddings a un formato que FAISS pueda usar
embeddings_cpu = embeddings.cpu().numpy().astype(np.float32)

# Crear un índice FAISS basado en la distancia L2 (distancia euclidiana)
index = faiss.IndexFlatL2(embeddings_cpu.shape[1])

# Añadir los vectores al índice
index.add(embeddings_cpu)

# Ver cuántos vectores están en el índice
print("Número de vectores en el índice:", index.ntotal)


# PASO 5 : Generar la query (preguntando al usurio)
query = input("Por favor, ingresa tu consulta: ")

# Vectorizar query 
embeddings_query = model.encode(query[32:], convert_to_tensor=True)

query_embedding_cpu = embeddings_query.cpu().numpy().astype(np.float32)

# Realizar la búsqueda en FAISS (buscar los k fragmentos más cercanos)
k = 5  # Número de fragmentos más relevantes que deseas recuperar
distances, indices = index.search(np.array(query_embedding_cpu.reshape(1, -1)), k)

query_prop = response_properties.QueryResults()

for i in range(0,k):
    #print(f'''Fragmento más cercano nº {i} : {divided_text[indices[0][i]]}''')
    query_prop.update_fields(i, distances[0][i], divided_text[indices[0][i]])
    print(f'''{query_prop.indice} | {query_prop.distance} | {query_prop.response}''')


# PASO 6 : Generar respuesta con gpt y verificar su relación con el resto de fragmentos cercanos