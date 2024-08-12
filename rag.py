from dotenv import load_dotenv
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings


def load_sections():
    all_sections = []
    with open("sections.txt", "r") as file:
        for line in file:
            section = line.strip()
            all_sections.append(section)
    return all_sections


def find_similar_contexts(sections, user_question, top_n=5):
    print("Finding similar contexts...")
    embedding = HuggingFaceEmbeddings(model_name = "oshizo/sbert-jsnli-luke-japanese-base-lite", encode_kwargs={"normalize_embeddings":True})
    vector_stores = FAISS.from_texts(sections, embedding)
    distances = vector_stores.similarity_search_with_score(user_question)
    sorted_distances = sorted(distances, key=lambda x: x[1])
    similar_contexts = sorted_distances[:top_n]
    print("Finished finding similar contexts")
    return similar_contexts


def concat_similar_contexts(similar_contexts):
    similar_contexts = [context[0].page_content for context in similar_contexts]
    similar_context = "\n".join(similar_contexts)
    return similar_context


def create_answer(context, user_question):
    vector_store = FAISS.from_texts([context], embedding=OpenAIEmbeddings())
    retriever = vector_store.as_retriever()

    template = '''
        以下の文脈のみに基づいて質問に答えなさい。日本語で答えなさい。: {context}
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
    sections = load_sections()
    similar_contexts = find_similar_contexts(sections, user_question)
    similar_context = concat_similar_contexts(similar_contexts)
    answer = create_answer(similar_context, user_question)
    print(answer)
