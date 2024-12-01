import os
import json
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough


client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def find_similar_contexts(query, top_n=10):
    query_embedding = get_embedding(query)
    
    embeddings_dir = "data/embeddings"
    texts_dir = "data/texts"
    all_embeddings = []
    file_indices = []
    file_mapping = {}
    
    for file in os.listdir(embeddings_dir):
        if file.endswith(".npy"):
            paper_id = os.path.splitext(file)[0]
            embeddings = np.load(os.path.join(embeddings_dir, file))
            all_embeddings.extend(embeddings)
            for i in range(len(embeddings)):
                file_indices.append(f"{paper_id}-{i}")
            with open(os.path.join(texts_dir, f"{paper_id}.txt"), "r") as text_file:
                lines = text_file.readlines()
                file_mapping[paper_id] = lines

    all_embeddings = np.array(all_embeddings)
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    top_indices = np.argsort(similarities)[-(top_n):][::-1]
    top_similarities = similarities[top_indices]
    
    similar_contexts = {}
    for idx, similarity in zip(top_indices, top_similarities):
        file_index = file_indices[idx]
        paper_id, line_index = file_index.split("-")
        line_index = int(line_index)
        context = file_mapping[paper_id][line_index].strip()
        similar_contexts[idx] = {"paper_id": paper_id, "context": context, "similarity": similarity}
    
    return similar_contexts


def get_title_and_author(paper_id):
    with open("data/mapping.json", 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        
    for paper in json_data:
        if paper['paper_id'] == paper_id:
            title = paper['title']
            author = paper['author']
            url = paper['url']
            return title, author, url
        

def create_context(similar_contexts):
    contexts = []
    for idx, context_info in similar_contexts.items():
        paper_id = context_info['paper_id']
        context = context_info['context']
        title, author, url = get_title_and_author(paper_id)
        context = f"Paper Title: {title}\nAuthor: {author}\nURL: {url}\n\nContext: {context}"
        contexts.append(context)
    return "\n\n\n".join(contexts)


def create_answer(context, user_question):
    vector_store = FAISS.from_texts([context], embedding=OpenAIEmbeddings())
    retriever = vector_store.as_retriever()

    template = '''
        質問内容が以下の文脈と関連のある場合、この内容のみに基づいて質問に答えなさい。
        回答するときは必ず、下記の論文情報を参照し、情報の元となる論文のタイトルと著者名、URLを記載しなさい。
        関連のない場合、論文には記載がなかったことを述べた上で、一般的な知識をもとに答えなさい。
        日本語で論理的に答えなさい。
        {context}
        質問: {question}
    '''
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain.invoke(user_question)


def return_rag_result(user_question):
    load_dotenv()
    similar_contexts = find_similar_contexts(user_question)
    context = create_context(similar_contexts)
    answer = create_answer(context, user_question)
    return answer
