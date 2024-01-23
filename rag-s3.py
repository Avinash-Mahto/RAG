## Install these packages

# Import modules
import openai
import chromadb
import os
import sys
import boto3
from io import BytesIO

# Import classes from modules
from langchain_community.document_loaders import S3DirectoryLoader
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAIEmbeddings


# Generate API KEY from OPENAI website and define as a variable. If you want to hide API key, you can use a constant.
os.environ["OPENAI_API_KEY"] = ""

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]


loader = S3DirectoryLoader("BUKCET_NAME", aws_access_key_id="ACCESS_KEY_ID", aws_secret_access_key="SECRET_KEY_ID")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# This part is used for embeddings the docs and store it into Vector DB and intialize the retriever.
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

# Create the RetrievalQA object
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

# This part is used to build a chat or Q&A application having the capability of both conversational capabilities and document retrieval
chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
)


chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])

  chat_history.append((query, result['answer']))
  query = None
