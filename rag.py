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


def find_similar_contexts(query, top_n=5):
    print(query)
    query_embedding = get_embedding(query)
    embeddings = np.load("data/embedded_paper.npy")
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-(top_n):][::-1]
    top_similarities = similarities[top_indices]
    with open("data/loaded_paper.txt", "r") as file:
        lines = file.readlines()
    similar_contexts = [(lines[index], similarity) for index, similarity in zip(top_indices, top_similarities)]
    return similar_contexts


def concat_similar_contexts(similar_contexts):
    similar_contexts = [context[0] for context in similar_contexts]
    similar_context = "\n".join(similar_contexts)
    return similar_context


def create_answer(context, user_question):
    vector_store = FAISS.from_texts([context], embedding=OpenAIEmbeddings())
    retriever = vector_store.as_retriever()

    template = '''
        質問内容が以下の文脈と関連のある場合、この文脈のみに基づいて質問に答えなさい。
        関連のない場合、論文には記載がなかったことを述べた上で、一般的な知識をもとに答えなさい。
        日本語で論理的に答えなさい。: {context}
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
    similar_context = concat_similar_contexts(similar_contexts)
    answer = create_answer(similar_context, user_question)
    return answer
