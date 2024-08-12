import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


def load_pdf_content(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


def create_sections_from_content(file_path):
    content = load_pdf_content(file_path)
    print("Creating sections...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
    )
    sections = text_splitter.split_documents(content)
    sections = [section.page_content for section in tqdm(sections)]
    print("Finished creating sections")
    return sections


def main():
    all_sections = []
    for filename in os.listdir("paper"):
        file_path = os.path.join("paper", filename)
        if os.path.isfile(file_path):
            sections = create_sections_from_content(file_path)
            all_sections.append(sections)
    with open("sections.txt", "w") as file:
        for section in all_sections:
            file.write(f"{section}\n")
    return


if __name__ == '__main__':
    main()