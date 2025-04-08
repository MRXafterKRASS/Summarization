import requests
import json
import re
import ollama
import html2text
from bs4 import BeautifulSoup
import os
import numpy as np
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

API_URL = "http://localhost:1245/v1/chat/completions"  # Замени XXXX на нужный порт
HEADERS = {"Content-Type": "application/json"}
OLLAMA_API = "http://localhost:11434/v1"

def fetch_and_convert_to_md(url, output_file="output.md"):
    try:
        # Загружаем страницу
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Разбираем HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Извлекаем заголовок и основной контент
        title = soup.title.string if soup.title else "Untitled"
        content = soup.body if soup.body else soup

        # Конвертируем в Markdown
        converter = html2text.HTML2Text()
        md_text = f"# {title}\n\n" + converter.handle(str(content))

        # Сохраняем в файл
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(md_text)
            print('страница переведена в md')
    except requests.RequestException as e:
        print(f"Ошибка при загрузке страницы: {e}")


def load_markdown(md_file):
    loader = UnstructuredMarkdownLoader(md_file)
    documents = loader.load()

    # Разбиваем текст на куски
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    return texts

class OllamaEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def create_faiss_index(texts):
    # Извлекаем тексты
    text_list = [doc.page_content for doc in texts]

    # Создаем объект эмбеддингов
    embedding_instance = OllamaEmbeddings()

    # Вычисляем эмбеддинги для каждого текста
    embeddings = [embedding_instance.embed_query(text) for text in text_list]

    # При необходимости преобразуем в numpy-массив
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Формируем список кортежей (текст, эмбеддинг)
    text_embeddings = list(zip(text_list, embeddings_array))

    # Передаем список кортежей и объект эмбеддингов в метод
    vectorstore = FAISS.from_embeddings(text_embeddings, embedding_instance)

    return vectorstore


# Поиск наиболее релевантных фрагментов
def retrieve_relevant_text(query, vectorstore, top_k=3):
    results = vectorstore.similarity_search(query, top_k)
    return "\n\n".join([r.page_content for r in results])


def send_request(prompt, model="local-model"):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Отвечай на русском языке."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }

    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        message = result["choices"][0]["message"]["content"]

        cleaned_message = re.sub(r"<think>.*?</think>\n?", "", message, flags=re.DOTALL).strip()

        return cleaned_message
    else:
        return f"Ошибка: {response.status_code}, {response.text}"

def rag_query(query, vectorstore, top_k=3):
    # Извлекаем релевантные тексты из векторного хранилища
    context = retrieve_relevant_text(query, vectorstore, top_k)
    # Формируем промпт, объединяя найденный контекст с исходным запросом
    combined_prompt = (
        f"Используй следующий контекст для ответа на вопрос.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {query}"
    )
    # Отправляем промпт в локальную модель для генерации ответа
    return send_request(combined_prompt)

if __name__ == "__main__":
    url = input("Введите URL: ")
    fetch_and_convert_to_md(url)
    md_file = "output.md"
    texts = load_markdown(md_file)
    vectorstore = create_faiss_index(texts)

    while True:
        user_input = input("Введи запрос (или 'exit' для выхода): ")
        if user_input.lower() == "exit":
            if os.path.exists('output.md'):
                os.remove('output.md')
            break
        response = rag_query(user_input,vectorstore)

        print("\nОтвет нейросети:\n", response, "\n")
