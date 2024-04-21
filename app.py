import streamlit as st
from PyPDF2 import PdfReader
import os
import torch
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()
st.set_page_config(layout="wide") 
pinecone_api_key=os.getenv("PINECONE_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


index_name="rag"
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 250, chunk_overlap = 20)

st.title("RAG based application")
uploaded_file = st.file_uploader("Upload a pdf file")

model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

if uploaded_file:

    reader=PdfReader(uploaded_file)
    condn=False

    def get_embeddings(text):
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            embeddings = model(input_ids)["last_hidden_state"].mean(dim=1)
            return embeddings.numpy()[0]
    
    # st.write(index.describe_index_stats)

    namespace_name=st.text_input("Enter your namespace?")
    if namespace_name:
        stats=index.describe_index_stats()
        namespaces=stats["namespaces"]
        if namespace_name in namespaces.keys():
            condn=True

        if not condn:

            total_text=[]
            for i in range(len(reader.pages)):
                page=reader.pages[i]
                texts=page.extract_text()
                texts=text_splitter.split_text(texts)
                docs=[]
                for text in texts:
                    docs.append(text)
                total_text+=docs

            vectorstore=[]

            for i in range(0, len(total_text)):
                res={}
                a=total_text[i]
                ascii_vector_id = a.encode('ascii', 'ignore').decode('ascii')
                res["id"]=ascii_vector_id
                res["values"]=get_embeddings(total_text[i]).tolist()
                vectorstore.append(res)


            def store_to_pinecone(vectorstore):
                index.upsert(
                    vectors=vectorstore,
                    namespace=namespace_name
                )
                st.write("Pdf is stored to the Vector Database")
                
            store_to_pinecone(vectorstore)

        query=st.text_input("Enter your query?")
        if query:
            query_embeddings=get_embeddings(query).tolist()

            res=index.query(
                vector=query_embeddings, top_k=5, 
                namespace=namespace_name
            )

            final=""
            for i in res["matches"]:
                final+=i["id"]
            
            # st.write(final)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(f'''
                User's Query - {query}
                Response received from Database - {final}
                You need to frame this response into complete sentences.
                Ensure that the exact meaning of response does not change and try to make only lil modifications in the response.
                Also if the query is present in the response, try to frame response as an answer of query but not displaying the query as it is.
            ''')

            st.write(response.text)
            # st.text_area("Results: ",value=final)


