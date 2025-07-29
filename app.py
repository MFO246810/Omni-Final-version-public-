import os
import uuid
import hashlib
import requests
import logging
import multiprocessing
from Models import Base
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from RAG_MODEL import RAG_MODEL
from Models import Active_Models
from sqlalchemy.orm import sessionmaker
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, select
from Models import Collections, User, Model_Collections, DataType

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    handlers=[
        logging.FileHandler("Omni.log"),
        logging.StreamHandler()
    ]
)

#ORM mapper for DB handling
engine = create_engine(os.getenv("DATABASE_URL"), echo=True)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()
Collection_Processes = []

@app.route('/Create_New_Model', methods=['POST'])
def Create_New_Model():
    #Expecting Models Name and the Agents prompt 
    Model_Name = request.form.get('ModelName')
    Model_Prompt = request.form.get('ModelAgent')
    User_ID = request.form.get('User_ID')
    if session.query(User).filter(User.User_ID == User_ID).count() == 0:
        app.logger.warning(f"User ID not found: {User_ID}")
        return jsonify({"Error": "User ID not found"}), 404
    app.logger.info(f"Creating new model with name: {Model_Name} and prompt: {Model_Prompt} for {User_ID}")
    if not Model_Name or not Model_Prompt or not User_ID:
        app.logger.warning("Model Name, Model Prompt, or User ID is missing")
        result = {
            "Status": "error",
            "Message": "Model Name, Model Prompt, or User ID is missing"
        }
        return jsonify(result), 400
    print(f"Model:{Model_Name}: {Model_Prompt}")
    Current_Models = session.query(Active_Models).all()
    if Model_Name in Current_Models:
        app.logger.warning("Model Name is in use")
        result = {
            "Status": "error",
            "Message": "Model name is already in use please use a different name"
        }
        return jsonify(result), 500
    else: 
        New_Model = Active_Models(Model_Name = Model_Name, Model_AgenticPrompt = Model_Prompt, Is_Active = False, User_ID=User_ID)
        session.add(New_Model)
        session.commit()
        result = {
            "Status": "Sucess",
            "Message": "Model has been created sucessfully"
        }
        return jsonify(result), 200
   
@app.route('/Handle_File_Data', methods=['POST'])
def Handle_File_Data():
    Data_Type = request.form.get("Data_Type")
    Model_Name = request.form.get("Model_Name")
    User_ID = request.form.get("User_ID")
    if session.query(User).filter(User.User_ID == User_ID).count() == 0:
        app.logger.warning(f"User ID not found: {User_ID}")
        return jsonify({"Error": "User ID not found"}), 404
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files.get('file')
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    Collection_Name = hash_file_content(file)
    Coll_stmt = select(Collections).where(Collections.Collections_Name == Collection_Name)
    Collection = session.scalars(Coll_stmt).first()
    Model_stmt = select(Active_Models).where(Active_Models.Model_Name == Model_Name)
    Current_Model = session.scalars(Model_stmt).first()
    print("Current Model: ", Current_Model)
    if Collection != None:
        stmt = select(Model_Collections).where(Model_Collections.Collection_Name == Collection_Name)
        Model_Check = session.scalars(stmt).first()
        if Model_Check == None or Model_Check.Model_ID == Current_Model.ID:
            app.logger.info(f"{Model.Model_Name} is adding data to collection: {Collection_name} with format: {format}")
            Model = RAG_MODEL(Model.Model_Name, Model.Model_AgenticPrompt)
            Model.MergeCollection(Collection_Name)
            app.logger.info(f"Collection {Collection_Name} has been merged with model: {Model.Model_Name}")
            return jsonify({"Message": "Collection is already added"}), 500
        else:
            Current_Model = RAG_MODEL(Model_Name, Current_Model.Model_AgenticPrompt)
            Model_Coll = Model_Collections(Model_ID=Current_Model.ID, Collection_Name=Collection_Name)
            session.add(Model_Coll)
            session.commit()
            app.logger.info(f"{Collection_Name} has been added to model: {Current_Model.ID}")
            return jsonify({"Message":"Dataset has been added to the model"}), 200
    else: 
        File_Directory = f'./Datasource/{Collection_Name}/'
        os.makedirs(File_Directory, exist_ok=True)
        Current_Model.Is_Active = True
        filepath = os.path.join(File_Directory, file.filename)
        file.save(filepath)
        if(Data_Type == "pdf"): 
            New_Data = Collections(Collections_Name=Collection_Name, Collections_Title=file.filename, Data_Type=DataType.pdf, Source_Original=filepath)
        elif(Data_Type == "epub"):
            New_Data = Collections(Collections_Name=Collection_Name, Collections_Title=file.filename, Data_Type=DataType.epub, Source_Original=filepath)
        else:
            New_Data = Collections(Collections_Name=Collection_Name, Collections_Title=file.filename, Data_Type=DataType.txt, Source_Original=filepath)
        Model_Coll = Model_Collections(Model_ID=Current_Model.ID, Collection_Name=Collection_Name)
        session.add(New_Data)
        session.add(Model_Coll)
        session.commit()
        app.logger.info(f"Creating new collection: {Collection_Name} with title: {file.filename}")
        app.logger.info(f"Starting process to add file data to collection: {Collection_Name}")
        print(Collection_Name)
        Collection_Process = multiprocessing.Process(target=Add_To_Collection, args=(filepath, Data_Type, Current_Model, Collection_Name))
        Collection_Process.start()
        Collection_Processes.append(Collection_Process)
        app.logger.info(f"Collection Process: {Collection_Process.pid} started for {Collection_Name}")
        app.logger.info(f"File data is being chunked and added to collection: {Collection_Name}")
        return jsonify({"message": f"Data is currently being chunked to {filepath}"}), 200

@app.route('/Handle_Webpage_Data', methods=['POST'])
def Handle_Webpage_Data():
    Model_Name = request.form.get("Model_Name")
    Webpage = request.form.get("Webpage")
    User_ID = request.form.get("User_ID")
    if session.query(User).filter(User.User_ID == User_ID).count() == 0:
        app.logger.warning(f"User ID not found: {User_ID}")
        return jsonify({"Error": "User ID not found"}), 404
    if not Model_Name or not Webpage:
        app.logger.warning("Model Name or Webpage is missing")
        return jsonify({"Error": "Model Name or Webpage is missing"}), 400
    if not Webpage.startswith(('http://', 'https://')):
        app.logger.warning("Webpage must be a valid URL")
        return jsonify({"Error": "Webpage must be a valid URL"}), 400
    if not requests.get(Webpage).ok:
        app.logger.warning(f"Webpage is not reachable: {Webpage}")
        return jsonify({"Error": "Webpage is not reachable"}), 400
    if not Model_Name:
        app.logger.warning("Model Name is missing")
        return jsonify({"Error": "Model Name is missing"}), 400
    Collection_Name = hash_url(Webpage)
    app.logger.info(f"Handling webpage data for Model: {Model_Name} and Webpage: {Webpage}")
    app.logger.info(f"Webpage data to be saved into this collection: {Collection_Name}")
    Coll_stmt = select(Collections).where(Collections.Collections_Name == Collection_Name)
    Collection = session.scalars(Coll_stmt).first()
    Model_stmt = select(Active_Models).where(Active_Models.Model_Name == Model_Name)
    Current_Model = session.scalars(Model_stmt).first()
    print("Current Model: ", Current_Model)
    if Collection != None:
        stmt = select(Model_Collections).where(Model_Collections.Collection_Name == Collection_Name)
        Model_Check = session.scalars(stmt).first()
        if Model_Check == None or Model_Check.Model_ID == Current_Model.ID:
            print("Collection has already beeen added")
            Model = RAG_MODEL(Model.Model_Name, Model.Model_AgenticPrompt)
            Model.MergeCollection(Collection_Name)
            app.logger.info(f"Collection {Collection_Name} has been merged with model: {Model.Model_Name}")
            return jsonify({"Error": "Collection is already added"}), 500
        else: 
            Model_Coll = Model_Collections(Model_ID=Current_Model.ID, Collection_Name=Collection_Name)
            session.add(Model_Coll)
            app.logger.info(f"{Collection_Name} has been added to model: {Current_Model.ID}")
            session.commit()
            return jsonify({"Message":"Dataset has been added to the model"}), 200
    else:
        response = requests.get(Webpage, timeout=30)
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title').text
        New_Data = Collections(Collections_Name=Collection_Name, Collections_Title=title, Data_Type=DataType.url, Source_Original=Webpage)
        Model_Coll = Model_Collections(Model_ID=Current_Model.ID, Collection_Name=Collection_Name)
        app.logger.info(f"Creating new collection: {Collection_Name} with title: {title}")
        session.add(New_Data)
        session.add(Model_Coll)
        session.commit()
        print(Collection_Name)
        app.logger.info(f"Starting process to add webpage data to collection: {Collection_Name}")
        Collection_Process = multiprocessing.Process(target=Add_To_Collection, args=(Webpage, "Webpage", Current_Model, Collection_Name))
        app.logger.info(f"Collection Process: {Collection_Process.pid} started for {Collection_Name}")
        Collection_Process.start()
        Collection_Processes.append(Collection_Process)
        return jsonify({"message": f"Data is currently being chunked"}), 200

@app.route('/Get_All_Models', methods=['POST'])
def Get_All_Models():
    User_Name = request.forms.get("User_Name")
    User_ID = request.form.get("User_ID")
    if session.query(User).filter(User.User_ID == User_ID).count() == 0:
        app.logger.warning(f"User ID not found: {User_ID}")
        return jsonify({"Error": "User ID not found"}), 404
    if not User_Name:
        return jsonify({"Message": "User Name is missing"}), 400
    app.logger.info(f"Fetching all the models for user: {User_Name}")
    User_stmt = select(User.User_ID).where(User.User_Name == User_Name)
    Current_User = session.scalars(User_stmt).first()
    stmt = select(Active_Models.Model_Name).where(Current_User.User_ID == Active_Models.User_ID)
    models = session.scalars(stmt).all()
    data = {
        "Current_Models": models
    }
    app.logger.info(f"Models fetched for user {User_Name}: {models}")
    return jsonify(data), 200

@app.route('/Create_User', methods=['POST'])
def Create_User():
    User_Name = request.form.get("User_Name")
    Password = request.form.get("Password")
    if not User_Name or not Password:
        return jsonify({"Message": "User Name or Password is missing"}), 400
    if len(User_Name) < 3 or len(Password) < 6:
        return jsonify({"Message": "User Name must be at least 3 characters and Password must be at least 6 characters"}), 400
    if not User_Name.isalnum():
        return jsonify({"Message": "User Name must be alphanumeric"}), 400
    if not Password.isalnum():
        return jsonify({"Message": "Password must be alphanumeric"}), 400
    if len(User_Name) > 20 or len(Password) > 20:
        return jsonify({"Message": "User Name and Password must be less than 20 characters"}), 400
    app.logger.info(f"Creating new user with name: {User_Name}")
    All_Users = session.query(User).all()
    for Current_User in All_Users:
        if(Current_User.User_Name == User_Name):
            return jsonify({"Message": "Username is already in use"}), 500
    New_User = User(User_Name=User_Name, Password= Password)
    session.add(New_User)
    session.commit()
    user_stmt = select(User).where(User.User_Name==User_Name)
    New_User = session.scalars(user_stmt).first()
    app.logger.info(f"New user created: {New_User.User_Name} with ID: {New_User.User_ID}")
    print(New_User)
    return jsonify({"Message": "User created sucessfully", "User": New_User.User_ID}), 200

@app.route('/Handle_User_Query', methods=['POST'])
def Handle_User_Query():
    Model_Name = request.form.get("Model_Name")
    User_Query = request.form.get("User_Query")
    User_ID = request.form.get("User_ID")
    if not User_ID:
        return jsonify({"Error": "User ID is missing"}), 400
    if not Model_Name or not User_Query:
        return jsonify({"Error": "Model Name or User Query is missing"}), 400
    app.logger.info(f"Handling user query for Model: {Model_Name} with Query: {User_Query}")
    if not Model_Name:
        return jsonify({"Error": "Model Name is missing"}), 400
    if not User_Query:
        return jsonify({"Error": "User Query is missing"}), 400
    if session.query(User).filter(User.User_ID == User_ID).count() == 0:
        app.logger.warning(f"User ID not found: {User_ID}")
        return jsonify({"Error": "User ID not found"}), 404
    Model_Call = select(Active_Models).where(Active_Models.Model_Name == Model_Name)
    Current_Model = session.scalars(Model_Call).first()
    if(Current_Model == None):
        app.logger.warning(f"Model not found: {Model_Name}")
        return jsonify({"Error": "Model could not be found"}), 404
    else: 
        if(Current_Model.Is_Active == False): 
            app.logger.warning(f"Model is not active: {Model_Name}")
            Get_Collection_stmt = select(Model_Collections.Collection_Name).where(Model_Collections.Model_ID == Current_Model.ID)
            Current_Collections = session.scalars(Get_Collection_stmt).all()
            print(Current_Collections)
            Current_Model = RAG_MODEL(Model_Name, Current_Model.Model_AgenticPrompt)
            app.logger.info(f"Handling user query: {User_Query} for model: {Model_Name}")
            result = Current_Model.HandleUserQuery(User_Query)
            print(result)
            return jsonify({"Response": result["Text"]}), 200
            #return jsonify({"Error": "No Data could be found"}), 500
        else: 
            Get_Collection_stmt = select(Model_Collections.Collection_Name).where(Model_Collections.Model_ID == Current_Model.ID)
            Current_Collections = session.scalars(Get_Collection_stmt).all()
            print(Current_Collections)
            Current_Model = RAG_MODEL(Model_Name, Current_Model.Model_AgenticPrompt)
            app.logger.info(f"Handling user query: {User_Query} for model: {Model_Name}")
            result = Current_Model.HandleUserQuery(User_Query)
            return jsonify({"Response": result["Text"]}), 200

@app.route('/Authenticate_User', methods=['POST'])
def Authenticate_User():
    User_Name = request.form.get("User_Name")
    Password = request.form.get("Password")
    if not User_Name or not Password:
        return jsonify({"Message": "User Name or Password is missing"}), 400
    stmt = select(User).where(User.User_Name == User_Name and User.Password == Password)
    Current_User = session.scalars(stmt).first()
    app.logger.info(f"Authenticating user: {User_Name}")
    if Current_User == None:
        return jsonify ({"Message": "User could not be verified please try agin"}), 404
    else:
        return jsonify ({"User": Current_User.User_ID}), 200

def Add_To_Collection(FilePath, format, Model, Collection_name):
    print("File Path: ", FilePath)
    app.logger.info(f"{Model.Model_Name} is adding data to collection: {Collection_name} with format: {format}")
    Current_Model = RAG_MODEL(Model.Model_Name, Model.Model_AgenticPrompt)
    app.logger.info(f"Creating a new collection: {Collection_name} for model: {Model.Model_Name}")
    Current_Model.CreateCollection(Collection_name)
    app.logger.info(f"Adding data to collection: {Collection_name} with format: {format}")
    Current_Model.AddToCollection(format, FilePath, Collection_name)
    return 0

def hash_file_content(file_obj) -> str:
    hasher = hashlib.sha256()
    for chunk in iter(lambda: file_obj.read(4096), b""):
        hasher.update(chunk)
    file_obj.seek(0)  # Reset file pointer after reading
    return hasher.hexdigest()  

def hash_url(url):
    return hashlib.sha256(url.encode()).hexdigest() 

if __name__ == '__main__':
    app.run(debug=True)