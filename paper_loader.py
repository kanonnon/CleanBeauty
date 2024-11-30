import os
import numpy as np
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


client = OpenAI()


def log_error(log_file_path, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")


def load_pdf_content(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


def create_sections_from_content(file_path):
    content = load_pdf_content(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    sections = text_splitter.split_documents(content)
    sections = [section.page_content for section in sections]
    return sections


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    output = client.embeddings.create(input = [text], model=model).data[0].embedding
    return output


def main():
    for filename in tqdm(os.listdir("data/papers")):
        embeddings = []
        file_path = os.path.join("data/papers", filename)
        file_id = filename.split(".")[0]
        if not file.endswith(".pdf"):
            continue
        try:
            sections = create_sections_from_content(file_path)
        except Exception as e:
            log_error("data/errors.txt", f"Error loading content from {file_path}: {e}")
            continue
        with open(f"data/texts/{file_id}.txt", "w") as file:
            for section in sections:
                section = section.replace("\n", "")
                file.write(f"{section}\n")
                try:
                    embeddings.append(get_embedding(section))
                except Exception as e:
                    log_error("data/errors.txt", f"Error embedding text from {file_path}: {e}")
                    continue
        np.save(f"data/embeddings/{file_id}.npy", np.array(embeddings))


if __name__ == '__main__':
    main()
