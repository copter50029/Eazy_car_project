from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader

pdf_path = "../data/EazyCar_Product_knowledge_for_AI_chatbot.pdf"
excel_path = "../data/Eazy Car_Package.xlsx"

pdf_loader = UnstructuredPDFLoader(pdf_path)
excel_loader = UnstructuredExcelLoader(excel_path)

pdf_docs = pdf_loader.load()
excel_docs = excel_loader.load()

docs = pdf_docs + excel_docs
print(docs[0].page_content)
# %pip install --q unstructured langchain langchain-community
# %pip install --q "unstructured[all-docs]" ipywidgets tqdm
# !ollama pull nomic-embed-text:v1.5
# !ollama list
# !pip install -q chromadb
# !pip install -q langchain-text-splitters
# !ollama pull deepseek-r1:1.5b
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chucks = text_splitter.split_documents(docs)
#Add to vector database
vector_db = Chroma.from_documents(
    documents=chucks,
    embedding=OllamaEmbeddings(model="nomic-embed-text:v1.5",show_progress=True),
    collection_name="Eazycar_db"
)
local_model = "deepseek-r1:1.5b"
llm = ChatOllama(model=local_model)
QUERRY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}
    """
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERRY_PROMPT
)

#RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
""" 
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# # chain.invoke(input(""))
# print("Ask a question about Eazy Car products: ")
# answer = chain.invoke(input(""))
# print("answer:", answer)
print("Welcome to Eazy Car product assistant!")
print("Ask a question about Eazy Car products: ")
while True:
    answer = chain.invoke(input(""))
    print("answer:", answer)
    if not answer:
        break