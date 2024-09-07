import os
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


client = OpenAI()


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
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def main():
    embeddings = []
    with open("data/loaded_paper.txt", "w") as file:
        for filename in tqdm(os.listdir("paper")):
            file_path = os.path.join("paper", filename)
            if os.path.isfile(file_path):
                sections = create_sections_from_content(file_path)
                
                for section in sections:
                    section = section.replace("\n", "")
                    file.write(f"{section}\n")
                    embeddings.append(get_embedding(section))
    np.save("data/embedded_paper.npy", np.array(embeddings))


if __name__ == '__main__':
    main()
