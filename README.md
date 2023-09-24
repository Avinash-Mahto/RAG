# Retrieval Augmented Generation (RAG)
We are building a Chatbot or Q&A application based on "Retrieval-based apporach". This approach is called RAG - Retrieval Augmented Generation.
Many LLM applications require user-specific data that is not part of the model's training set. The primary way of accomplishing this is through Retrieval Augmented Generation (RAG). 

# Install following packages
pip install openai langchain tiktoken chromadb untructured

# Create your own dataset under Directory , as I keep my custom data under directory - /mydata (directory name could be anything)
if you have text data then use only "textloader", if you have pdf file, use pdfloader.

