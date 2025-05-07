# Libreria de python necesarias para el RAG
#import langchain 
#import faiss
#import openai
#import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import utils

# PASO 1 : Leer el archivo
path_pdf = './APUNTES_HISTORIA_ESPAN_A._2_BCHTO__2018-19.pdf'

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

# PASO 6 : Generar respuesta con gpt