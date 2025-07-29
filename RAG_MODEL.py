import os
import fitz
import time
import uuid
import requests
import chromadb
import logging
import urllib.request
from PIL import Image
from io import BytesIO
from google import genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from transformers import AutoTokenizer
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
import PIL.Image
load_dotenv()

class RAG_MODEL:
    def __init__(self, name, Agent_Prompt):
        self.name = name
        self.Agent_Prompt = Agent_Prompt
        self.ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ChosenCollection = None
        self.chroma_client = None
        if(os.path.exists(f"./datasets")):
            pass
        else: 
            os.makedirs(f"./datasets")
        self.client_dir = f"./datasets"
        self.image_dir = None
        self.chroma_client = chromadb.PersistentClient(path=self.client_dir)
        self.Model =self.chroma_client.get_or_create_collection(name=name)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def CreateCollection(self, CollectionName):
        self.image_dir = f"./Datasource/{CollectionName}/Images/"
        self.ChosenCollection = self.chroma_client.get_or_create_collection(name=CollectionName)

    def AddToCollection(self, format, datasource, collection_name):
        if(collection_name == None):
            return None
        else: 
            self.ChosenCollection = self.chroma_client.get_or_create_collection(name=collection_name)
            if(format == "Webpage"):
                text_and_metadata = []
                text = {}
                metadata = {}
                print(f"Webpage: {datasource}")
                self.logger.info(f"Processing webpage: {datasource}")
                response = requests.get(datasource, timeout=30)
                soup = BeautifulSoup(response.content, 'html.parser')
                text["data"] = soup.get_text(strip=True)
                title = soup.find('title').text
                meta_description = soup.find('meta', attrs={'name': 'description'})
                if meta_description:
                    description = meta_description['content']
                else:
                    description = "None"
                text['image'] = RAG_MODEL.extract_and_downloads(self, response.content, datasource, image_folder=self.image_dir)
                self.logger.info(f"Extracted images from webpage: {datasource}")
                metadata["title"] = title
                metadata["description"] = description
                metadata["Uri"] = datasource
                self.logger.info(f"Metadata for webpage: {metadata}")
                text["metadata"] = metadata
                text_and_metadata.append(text)
                RAG_MODEL.save_to_db(self, format, RAG_MODEL.Embedded_Chunks(self, RAG_MODEL.chunker(self, text_and_metadata)))
                #Parse Html here with beautiful soup
                print(f"{title} webpage has been added to the collection")
            elif(format == "epub"):
                print("Epub")
                doc = fitz.open(datasource)
                text_and_metadata = []
                metadata = {}
                metadata['format']= doc.metadata['format']
                metadata['title'] = doc.metadata['title']
                metadata['author'] = doc.metadata['author']
                metadata['subject'] = doc.metadata['subject']
                metadata['creator'] = doc.metadata['creator']
                metadata['producer'] = doc.metadata['producer']
                metadata['creationDate'] = doc.metadata['creationDate']
                self.logger.info(f"Processing epub: {datasource}")
                for i in range(doc.chapter_count):
                    chapter_page_count = doc.chapter_page_count(i)
                    text = {}
                    for j in range(chapter_page_count):
                        page = doc[(i, j)]
                        text['data'] = page.get_text()
                        text['image'] = RAG_MODEL.extract_and_download_pdf(self, page, i, doc=doc, imageuri=self.image_dir)
                        self.logger.info(f"Extracted images from epub chapter {i}, page {j}")
                        metadata['CurrentChapter'] = i
                        metadata['pagenumber'] = j
                        text['metadata'] = metadata
                        text_and_metadata.append(text)
                RAG_MODEL.save_to_db(self, format, RAG_MODEL.Embedded_Chunks(self, RAG_MODEL.chunker(self, text_and_metadata)))
                print(f"{title} file has been added to the collection")
                #Parse EPUB here with fitz 
            elif(format == "pdf"):
                print("PDF")
                text_and_metadata = []
                metadata = {}
                doc = fitz.open(datasource)
                metadata['format']= doc.metadata['format']
                metadata['title'] = doc.metadata['title']
                metadata['author'] = doc.metadata['author']
                metadata['subject'] = doc.metadata['subject']
                metadata['creator'] = doc.metadata['creator']
                metadata['producer'] = doc.metadata['producer']
                metadata['creationDate'] = doc.metadata['creationDate']
                self.logger.info(f"Processing pdf: {datasource}")
                text = {}
                for i in range(doc.page_count):
                    page = doc.load_page(i)
                    text['data'] = page.get_text()
                    metadata['pagenumber'] = i
                    text['metadata'] = metadata
                    text['image'] = RAG_MODEL.extract_and_download_pdf(self, page, i, doc=doc, imageuri=self.image_dir)
                    text_and_metadata.append(text)
                    text = {}
                RAG_MODEL.save_to_db(self, format, RAG_MODEL.Embedded_Chunks(self, RAG_MODEL.chunker(self, text_and_metadata)))
                print(f"{title} file has been added to the collection")
                #Parse PDF here with pymupdf 
            elif(format == "txt"):
                print("txt: ")
                text_and_metadata = []
                metadata = {}
                text = {}
                with open(datasource, 'r') as file:
                    content = file.read()
                    text['data'] = content
                    metadata['format'] = "txt"
                    FileStat = os.stat(datasource)
                    metadata['title'] = os.path.splitext(os.path.basename(datasource))[0]
                    metadata['creation_time'] = time.ctime(FileStat.st_ctime)
                    metadata['Modified_time'] = time.ctime(FileStat.st_mtime)
                    text['metadata'] = metadata
                    text_and_metadata.insert(text)
                self.logger.info(f"Processing txt file: {datasource}")
                self.logger.info(f"Metadata for txt file: {metadata}")
                RAG_MODEL.save_to_db(self, format, RAG_MODEL.Embedded_Chunks(self, RAG_MODEL.chunker(self, text_and_metadata)))
                print(f"{title} file has been added to the collection")
            #parse txt with the file library
            else:
                print("Not an acceptable format")

    def chunker(self, BigChunks, max_tokens=512, overlap=50):
        self.logger.info("Chunking data into smaller pieces")
        result = []
        if(len(BigChunks) == 0):
            self.logger.warning("No data to chunk")
            return None
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        for chunks in BigChunks:
            tokens = tokenizer.encode(chunks['data'], add_special_tokens=False)
            chunk = {}
            for i in range(0, len(tokens), max_tokens - overlap):
                chunk_tokens = tokens[i:i + max_tokens]
                chunk['data'] = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunk['metadata'] = chunks['metadata']
                chunk['image'] = chunks['image']
                result.append(chunk)
        self.logger.info(f"Chunking completed, {len(result)} chunks created")
        if(len(result) == 0):
            self.logger.warning("No chunks created, check the input data")
            return None
        return result

    def save_to_db(self, format, Embedded_chunks):
        self.logger.info(f"Saving {len(Embedded_chunks)} chunks to the database")
        if(self.ChosenCollection == None):
            self.logger.error("No collection chosen, cannot save to database")
            return None
        if(len(Embedded_chunks) == 0):
            self.logger.warning("No data to chunk, skipping save to database")
            return None
        if(format == "Webpage"):
            self.logger.info("Saving webpage chunks to the database")
            for chunk in Embedded_chunks:
                chunk['metadata'] = {k: (v if v is not None else "") for k, v in chunk['metadata'].items()}
                if(chunk['image'] == None):
                    chunk['metadata']['img_size'] = 0
                    pass
                else:
                    for i, img in enumerate(chunk['image']): 
                        chunk['metadata'][f'img_src{i}'] = img['src']
                        chunk['metadata'][f'img_filepath{i}'] = img['filepath'] 
                        if(len(chunk['image']) > 0):
                            chunk['metadata']['img_size'] = len(chunk['image'])
                        else:
                            chunk['metadata']['img_size'] = 0

                    self.ChosenCollection.upsert(
                        documents=[chunk['data']],  
                        ids=[str(uuid.uuid4())],  
                        embeddings=[chunk['embeddings']],
                        metadatas=[chunk['metadata']]) 

                    self.Model.upsert(
                        documents=[chunk['data']],  
                        ids=[str(uuid.uuid4())],  
                        embeddings=[chunk['embeddings']],
                        metadatas=[chunk['metadata']]
                    )  
        elif(format in ['pdf','epub']):
            self.logger.info(f"Saving {format} chunks to the database")
            for chunk in Embedded_chunks:
                chunk['metadata'] = {k: (v if v is not None else "") for k, v in chunk['metadata'].items()}
                if(chunk['image'] == None):
                    chunk['metadata']['img_size'] = 0
                    pass
                else:
                    for i, img in enumerate(chunk['image']): 
                        chunk['metadata'][f'img_filepath{i}'] = img
                        if(len(chunk['image']) > 0):
                            chunk['metadata']['img_size'] = len(chunk['image'])
                        else:
                            chunk['metadata']['img_size'] = 0

                    self.ChosenCollection.upsert(
                        documents=[chunk['data']],  
                        ids=[str(uuid.uuid4())],  
                        embeddings=[chunk['embeddings']],
                        metadatas=[chunk['metadata']])
                    
                    self.Model.upsert(
                        documents=[chunk['data']],  
                        ids=[str(uuid.uuid4())],  
                        embeddings=[chunk['embeddings']],
                        metadatas=[chunk['metadata']]
                    )  
        elif(format == 'txt'):
            self.logger.info("Saving txt chunks to the database")
            for chunk in Embedded_chunks:
                chunk['metadata'] = {k: (v if v is not None else "") for k, v in chunk['metadata'].items()}
                self.ChosenCollection.upsert(
                    documents=[chunk['data']],  
                    ids=[str(uuid.uuid4())],  
                    embeddings=[chunk['embeddings']],
                    metadatas=[chunk['metadata']])
                
                self.Model.upsert(
                    documents=[chunk['data']],  
                    ids=[str(uuid.uuid4())],  
                    embeddings=[chunk['embeddings']],
                    metadatas=[chunk['metadata']]
                )  

    def extract_and_downloads(self, html_content, base_url=None, image_folder="Image_Folder"):
        self.logger.info("Extracting and downloading images from HTML content")
        soup = BeautifulSoup(html_content, 'html.parser')
        os.makedirs(image_folder, exist_ok=True)
        images_data = []

        for img in soup.find_all('img'):
            filepath = None
            src = img.get('src')
            if not src:
                continue

            # Handle relative URLs
            if base_url:
                src = urllib.parse.urljoin(base_url, src)

            alt_text = img.get('alt', '')
            title_text = img.get('title', '')
            caption = ""

            # Try to find figcaption if the <img> is inside a <figure>
            if img.parent.name == "figure":
                figcaption = img.parent.find("figcaption")
                if figcaption:
                    caption = figcaption.text.strip()

            # Fallback to paragraph context
            if not caption:
                para = img.find_parent("p")
                if para:
                    caption = para.text.strip()

            if(src):
                response = requests.get(src)
                parsed = urlparse(src)
                filename = os.path.basename(parsed.path)
                filepath = os.path.join(image_folder, filename)

                with open(filepath, 'wb') as f:
                    f.write(response.content)

            if(alt_text == ''):
                alt_text = 'None'
            if(title_text == ''): 
                title_text = "None"
            if(caption == ''): 
                caption = "None"
            images_data.append({
                "src": src,
                "alt": alt_text,
                "title": title_text,
                "caption": caption,
                "filepath": filepath
            })
        self.logger.info(f"Extracted {len(images_data)} images from HTML content")
        if(len(images_data) == 0):
            self.logger.info("No images found in the HTML content")
            return None
        return images_data
    
    def extract_and_download_pdf(self,page, index, doc, imageuri):
        self.logger.info(f"Extracting and downloading images from PDF page {index + 1}")
        image_list = page.get_images(full=True)
        os.makedirs(imageuri, exist_ok=True)
        result = []
        if image_list:
            pass
        else:
            return None
        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_name = f"page_{index + 1}_image_{image_index}.{image_ext}"
            with open(f"{imageuri}/{image_name}", "wb") as image_file:
                image_file.write(image_bytes)
            result.append(f"{imageuri}/{image_name}")
        self.logger.info(f"Extracted {len(result)} images from PDF page {index + 1}")
        return result

    def Embedded_Chunks(self, token_chunks):
        self.logger.info("Embedding chunks of data")
        if(self.embedding_model == None):
            self.logger.error("Embedding model is not initialized")
            return None
        if(token_chunks == None):
            self.logger.warning("No token chunks provided, returning None")
            return None
        if(type(token_chunks) != list):
            self.logger.error("Token chunks should be a list, returning None")
            return None
        self.logger.info(f"Number of token chunks to embed: {len(token_chunks)}")
        if(len(token_chunks) == 0):
            self.logger.warning("No data to embed, returning None")
            return None
        result = []
        embedding_chunk = {}
        for chunks in token_chunks:
            embedding_chunk['data'] = chunks['data']
            embedding_chunk['metadata'] = chunks['metadata']
            embedding_chunk['image'] = chunks['image']
            embedding_chunk['embeddings'] = self.embedding_model.encode(chunks['data'])
            result.append(embedding_chunk)
            embedding_chunk = {}
        self.logger.info(f"Embedding completed, {len(result)} chunks embedded")
        return result

    def Merge_With_Model(self, Collection_Name: str):
        self.logger.info(f"Merging collection {Collection_Name} with the model {self.name}")
        All_Collections = self.chroma_client.list_collections()
        if(Collection_Name in All_Collections):
            self.Model.upsert(
                documents=Collection_Name.get()["documents"],  
                ids=Collection_Name.get()["ids"],  
                embeddings=Collection_Name.get()["embeddings"],
                metadatas=Collection_Name.get()["metadatas"]
            )
            self.logger.info(f"Collection {Collection_Name} merged with the model {self.name}")
        else: 
            pass
            self.logger.error(f"Collection {Collection_Name} not found, cannot merge with the model {self.name}")     

    def HandleUserQuery(self, query:str):
        embedded_query = self.embedding_model.encode(query)
        self.logger.info(f"Handling user query: {query}")
        if self.Model is None:
            self.logger.error("Model is not initialized, cannot handle user query")
            return None
        if embedded_query is None:
            self.logger.error("Embedded query is None, cannot handle user query")
            return None
        if len(embedded_query) == 0:
            self.logger.error("Embedded query is empty, cannot handle user query")
            return None
        self.logger.info(f"Querying the model {self.name} with the embedded query")
        results = self.Model.query(
            query_embeddings=[embedded_query],
            n_results=25
        )
        if self.Agent_Prompt == "":
            self.logger.info("Using default prompt for the model")
            base_prompt = {
            "content": "I am going to ask you a question, which I would like you to answer"
            " based only on the provided context, and not any other information."
            " If there is not enough information in the context to answer the question,"
            ' say "I am not sure", then try to make a guess.'
            " Break your answer up into nicely readable paragraphs.",
        }
        else: 
            self.logger.info("Using custom agent prompt for the model")
            base_prompt = {"content": self.Agent_Prompt}
            
        data = {
            "documents": results["documents"],
            "metadatas": results["metadatas"],
            "ids": results["ids"]
        }

        user_prompt = {
            "content": f" The question is '{query}'. Here is all the context you have:"f"{data['documents']}",
        }
        images = []
        images_path = []
        seen =[]
        for metadata in data['metadatas']:
            for meta in metadata:
                for i in range(int(meta['img_size'])):
                    if(meta[f'img_filepath{i}'] in seen):
                        pass
                    else:
                        images_path.append(meta[f'img_filepath{i}']) 
                    seen.append(meta[f'img_filepath{i}'])    
                for path in images_path:
                    _, ext = os.path.splitext(path)
                    if(ext.lower() == '.svg'):
                        pass
                    else:
                        images.append(PIL.Image.open(path))
        if(len(images) == 0):
            self.logger.info("No images found in the context, generating response without images")
            response = self.ai_client.models.generate_content(
            model="gemini-2.5-flash", contents=f"{base_prompt['content']} + {user_prompt['content']}",
        ) 
        else:
            self.logger.info(f"Found {len(images)} images in the context, generating response with images")
            response = self.ai_client.models.generate_content(
            model="gemini-2.5-flash", contents=f"{base_prompt['content']} + photos from the context {images} + {user_prompt['content']}",
        )
        Query_Result = {
            "Text":response.text,
            "Metadata": data['metadatas']
        }
        self.logger.info(f"User query handled successfully, response generated")
        if(response.text == None):
            self.logger.error("Response text is None, returning None")
            return None
        print("Response Text: ", response.text)
        return Query_Result