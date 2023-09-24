## Install these packages
# pip install openai langchain tiktoken chromadb untructured

# Import modules
import openai
import os
import sys

#import class from modules

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

# Generate API KEY from OPENAI website and define as a variable. if you want to hide API key just import "constant"
# and define API key as constant.APIKEY  
os.environ["OPENAI_API_KEY"] = ""


# This function is used to pass the argument with query.

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]


#Load the custom Dataset and split into chunks, You can load data as (pdf, text file, html file and WebBaseLoader.)
loader = DirectoryLoader("mydata/")
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

#Initialize an empty list called chat_history to store the conversation history.
#Start a while loop that continues until the user enters "quit", "q", or "exit".
#Check if the variable query is empty. If it is, prompt the user to enter a query.
#Check if the user wants to quit the app. If the query matches "quit", "q", or "exit", exit the program.
#Call the chain object with the user's query and the current chat_history as input. This will generate a response from the app.
#Print the answer from the result dictionary.
#Append the user's query and the app's answer to the chat_history list.
#Repeat the loop to allow the user to enter another query. 

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
