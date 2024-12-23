# --- Carga de documentos
import os
import requests
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader

# Carga las variables de entorno desde el archivo especificado para obtener la API KEY de OpenAI.
load_dotenv("../secret/keys.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Establece la API KEY de OpenAI como variable de entorno para su uso posterior en el script.
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Lista de URLs desde donde se descargarán documentos en formato PDF.
urls = [
    'https://arxiv.org/pdf/2306.06031v1.pdf',
    'https://arxiv.org/pdf/2306.12156v1.pdf',
    'https://arxiv.org/pdf/2306.14289v1.pdf',
    'https://arxiv.org/pdf/2305.10973v1.pdf',
    'https://arxiv.org/pdf/2306.13643v1.pdf'
]

# Inicializa una lista vacía para almacenar los contenidos de los documentos.
ml_papers = []

# Bucle para descargar y cargar documentos desde las URLs especificadas.
for i, url in enumerate(urls):
    filename = f'paper{i+1}.pdf'

    # Descarga el archivo solo si no existe ya en el directorio actual.
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f'Descargado {filename}')
    else:
        print(f'{filename} ya existe, cargando desde el disco.')

    # Carga el documento PDF y almacena su contenido en la lista ml_papers.
    loader = PyPDFLoader(filename)
    data = loader.load()
    ml_papers.extend(data)

# Muestra el tipo y la longitud de la lista ml_papers y un ejemplo de contenido.
print('Contenido de ml_papers:')
print()
print(type(ml_papers), len(ml_papers), ml_papers[3])

# --- Particionado de documentos

# Importa el módulo necesario para particionar texto.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configura el particionador de texto para dividir los documentos en fragmentos de 1500 palabras con 200 palabras de solapamiento.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len
)

# Particiona los documentos y almacena el resultado en 'documents'.
documents = text_splitter.split_documents(ml_papers)
# Muestra la cantidad de documentos particionados y un ejemplo.
print(len(documents), documents[10])

# --- Embeddings e ingesta a base de datos vectorial

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Crea un objeto de embeddings que convierte texto a vectores utilizando el modelo ADA-002 de OpenAI.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Crea un objeto para almacenar y recuperar los vectores de texto generados a partir de los documentos.
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

# Configura el objeto para recuperar los vectores más similares a una consulta con un límite de 3 resultados.
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# --- Modelos de Chat y cadenas para consulta de información

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Configura un modelo de chat utilizando GPT-3.5 con respuestas precisas y sin generación creativa.
chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

# Crea una cadena de consulta de preguntas y respuestas utilizando el chat y el vector de recuperación configurado.
qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever
)

# Ejecuta consultas de prueba y muestra las respuestas.
query = "qué es fingpt?"
print(query)
print(qa_chain.run(query))

query = "qué hace complicado entrenar un modelo como el fingpt?"
print(query)
print(qa_chain.run(query))

query = "qué es fast segment?"
print(query)
print(qa_chain.run(query))

query = "cuál es la diferencia entre fast sam y mobile sam?"
print(query)
print(qa_chain.run(query))
