import json
import os
import sqlite3
import cv2
from deepface import DeepFace
from deepface.modules import verification
import numpy as np
import pickle

# tables structure here : https://docs.google.com/document/d/1PrIiOg6oL_UbHzmdwUrBh9yQDj1p-Xr8mwjz3YOX8Bk/edit?usp=sharing
thresholds = {
    # "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86}, # 2622d
    "VGG-Face": {"cosine": 0.68, "euclidean": 1.17, "euclidean_l2": 1.17,},  # 4096d - tuned with LFW
    "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
    "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
    "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
    "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
    "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
    "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
    "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
    "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
    "GhostFaceNet": {"cosine": 0.65, "euclidean": 35.71, "euclidean_l2": 1.10},
    }
models = ["VGG-Face", "Facenet", "Facenet512", "ArcFace", "Dlib", "SFace", "OpenFace", "DeepFace", "DeepID", "GhostFaceNet"]

def get_unmapped_media_paths():
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()
    
    cursor.execute('''
                   SELECT name FROM sqlite_master WHERE type='table' AND name='facesUnprocessedMediaItems'
                   ''')
    table_exists = cursor.fetchone() is not None
    
    unmapped_media = []
    if table_exists:
        cursor.execute('SELECT * FROM facesUnprocessedMediaItems')
        unmapped_media = cursor.fetchall()  # Fetch both the file path and its type (image/movie)
    
    conn.close()
    return unmapped_media

def extract_frames_from_movie(movie_path, interval=5):
    video = cv2.VideoCapture(movie_path)
    frames = []
    frame_count = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break
        
        if frame_count % interval == 0:
            frames.append(frame, frame_count)
        
        frame_count += 1
    
    video.release()
    return frames
def testEmbeddingsVsVerify():
    img1 = "test/20240206_120433_face1.jpg"
    img2 = "test/20240206_120433_face2.jpg"
    v = DeepFace.verify(img1_path=img1, img2_path=img2, model_name="VGG-Face", distance_metric="cosine", detector_backend="opencv")
    print('v[Threshold] ', v['threshold'])
    print('v[distance] ', v['distance'])
    print(v["verified"])
    e1 = DeepFace.represent(img_path=img1)
    e2 = DeepFace.represent(img_path=img2)
    distance = verification.find_cosine_distance(e1[0]['embedding'], e2[0]['embedding'])
    print("verification.find_cosine_distance: ", distance)

    oneImage2Faces = "test/20240206_120433.jpg"
    embeddings_data = DeepFace.represent(img_path=oneImage2Faces, model_name="VGG-Face", enforce_detection=False)
    embeddings = []
    for data in embeddings_data:
        embedding = np.array(data['embedding'], dtype=np.float32)
        embeddings.append(embedding)
    print ("len(embeddings) : " + str(len(embeddings)))   
    distance = verification.find_cosine_distance(embeddings[0], embeddings[1])
    print("oneImage2Faces verification.find_cosine_distance: ", distance)

def test2FacesImage():
    oneImage2FacesPath = "test/20240206_120433.jpg"
    oneImage2FacesImg = cv2.imread(oneImage2FacesPath)
    #extract faces from image
    oneImage2FacesEmbeddings_data = DeepFace.represent(img_path=oneImage2FacesPath, model_name="ArcFace", enforce_detection=False)
    faces = []
    oneImage2FacesEmbeddings = []
    for data in oneImage2FacesEmbeddings_data:
        embeddingNp = np.array(data['embedding'], dtype=np.float32)
        embedding = data['embedding']
        oneImage2FacesEmbeddings.append(embedding)

        facial_area = data['facial_area']
        # Extract facial area details from the dictionary
        x = facial_area.get("x")
        y = facial_area.get("y")
        w = facial_area.get("w")
        h = facial_area.get("h")
        face = oneImage2FacesImg[y:y+h, x:x+w]
        faces.append(face)
    print ("***one image 2 faces*** : ")
    print ("len(oneImage2FacesEmbeddings) : " + str(len(oneImage2FacesEmbeddings)))
    distance = verification.find_cosine_distance(oneImage2FacesEmbeddings[0], oneImage2FacesEmbeddings[1])
    print("oneImage2FacesEmbeddings verification.find_cosine_distance: ", distance)

    print("***two images each single face : *** ")
    print ("len(faces) : " + str(len(faces)))
    two_images_verify = DeepFace.verify(img1_path=faces[0], img2_path=faces[1], model_name="ArcFace", distance_metric="cosine", detector_backend="opencv")
    print('v[Threshold] ', two_images_verify['threshold'])
    print('v[distance] ', two_images_verify['distance'])
    print("two_images_verify[verified]" + str(two_images_verify["verified"]))
    e1 = DeepFace.represent(img_path=faces[0])
    e2 = DeepFace.represent(img_path=faces[1])
    distance = verification.find_cosine_distance(e1[0]['embedding'], e2[0]['embedding'])
    print("two images each with one face verification.find_cosine_distance: ", distance)


def process_new_image(cursor, image, image_path, conn):
    # Get embeddings from DeepFace
    model_name = "Dlib"
    embeddings_data = DeepFace.represent(img_path=image, model_name=model_name, enforce_detection=False, detector_backend="retinaface")
    
    for data in embeddings_data:
        embedding = data['embedding']
        facial_area = data['facial_area']
        
        # Fetch all existing face groups from the database
        cursor.execute("SELECT personID, averageEmbedding, embeddingCount, imagePaths, facialArea FROM FaceGroups")
        rows = cursor.fetchall()

        distances = []
        for row in rows:
            stored_embedding = np.frombuffer(row[1], dtype=np.float64)
            distance = verification.find_cosine_distance(stored_embedding, embedding)
            distances.append(distance)
        
        if distances:
            min_distance_idx = np.argmin(distances)
            threshold = thresholds[model_name]['cosine']
            if distances[min_distance_idx] < threshold - 0.01: #for dlib 0.069 worked better then 0.07  # Adjust threshold as needed
                # Update existing person
                personID = rows[min_distance_idx][0]
                current_avg_embedding = np.frombuffer(rows[min_distance_idx][1], dtype=np.float64)
                num_embeddings = rows[min_distance_idx][2]
                
                # Calculate the new average embedding
                new_average_embedding = (current_avg_embedding * num_embeddings + embedding) / (num_embeddings + 1)
                
                # Retrieve and update the image paths and facial areas
                image_paths = json.loads(rows[min_distance_idx][3])
                facial_areas = json.loads(rows[min_distance_idx][4])
                
                if image_path not in image_paths:
                    image_paths.append(image_path)
                facial_areas.append(facial_area)
                
                # Calculate the unique image path count
                unique_image_path_count = len(set(image_paths))
                
                # Update the database with the new information
                cursor.execute("UPDATE FaceGroups SET averageEmbedding=?, imagePaths=?, facialArea=?, embeddingCount=?, uniqueImagePathCount=? WHERE personID=?", 
                               (new_average_embedding.tobytes(), json.dumps(image_paths), json.dumps(facial_areas), num_embeddings + 1, unique_image_path_count, personID))
            else:
                # Insert as a new person
                cursor.execute("INSERT INTO FaceGroups (averageEmbedding, imagePaths, facialArea, embeddingCount, uniqueImagePathCount) VALUES (?, ?, ?, ?, ?)", 
                               (np.array(embedding, dtype=np.float64).tobytes(), json.dumps([image_path] if image_path else []), json.dumps([facial_area]), 1, 1))
        else:
            # Insert as the first person
            cursor.execute("INSERT INTO FaceGroups (averageEmbedding, imagePaths, facialArea, embeddingCount, uniqueImagePathCount) VALUES (?, ?, ?, ?, ?)", 
                           (np.array(embedding, dtype=np.float64).tobytes(), json.dumps([image_path] if image_path else []), json.dumps([facial_area]), 1, 1))
        
        # Commit the changes to the database
        conn.commit()


def clusterEmbeddings(cursor, conn):
    cursor.execute("SELECT * FROM face_embeddings WHERE PERSONID IS NULL")
    face_embeddings_rows = cursor.fetchall()
    model_name = "Dlib"
    for row in face_embeddings_rows:
        # Deserialize the stored binary data back into a numpy array
        face_embedding = pickle.loads(row["embedding"])
        cursor.execute("SELECT personID, avgEmbedding, embeddingCount from Persons Table ")
        personsID_rows = cursor.fetchall()

        distances = []
        for personID_row in personsID_rows:
            personAvgEmbedding = pickle.loads(personID_row["avgEmbedding"]) #  todo store it with pickle 
            personID = personID_row["personID"]
            embeddingCount = personID_row["embeddingCount"]
            distance = verification.find_cosine_distance(personAvgEmbedding, face_embedding)
            distances.append((personID, distance, embeddingCount))
        
        if distances:
            min_distance_idx = min(enumerate(distances), key=lambda x: x[1][1])[0]
            threshold = thresholds[model_name]['cosine']
            if distances[min_distance_idx][1] < threshold - 0.01: #for dlib 0.069 worked better then 0.07  # Adjust threshold as needed
                #update person table
                personID = distances[min_distance_idx][0]
                personAvgEmbedding = personsID_rows[min_distance_idx][1]
                oldEmbeddingsCount = personsID_rows[min_distance_idx][2]
                newEmbeddingsCount = oldEmbeddingsCount + 1
                newAverageEmbedding = (personAvgEmbedding * oldEmbeddingsCount + face_embedding) / (newEmbeddingsCount)
                serialized_average_embedding = pickle.dumps(newAverageEmbedding)

                cursor.execute("UPDATE Persons SET avgEmbedding=?, embeddingCount=? WHERE personID=?", 
                               (serialized_average_embedding, newEmbeddingsCount, personID))
                
                # Update the personID in the face_embeddings table
                cursor.execute("UPDATE face_embeddings SET personID=? WHERE embeddingID=?",
                               (personID, row["embeddingID"]))
            else:
                #Insert as a new person
                cursor.execute("INSERT INTO Persons (avgEmbedding, embeddingCount) VALUES (?, ?)", 
                               (pickle.dumps(face_embedding), 1))
                
                # Insert the new personID in the face_embeddings table
                personID = cursor.lastrowid
                cursor.execute("UPDATE face_embeddings SET personID=? WHERE embeddingID=?",
                               (personID, row["embeddingID"]))
        else:
            #Insert as the first person
            cursor.execute("INSERT INTO Persons (avgEmbedding, embeddingCount) VALUES (?, ?)", 
                           (pickle.dumps(face_embedding), 1))
            
            # Insert the new personID in the face_embeddings table
            personID = cursor.lastrowid
            cursor.execute("UPDATE face_embeddings SET personID=? WHERE embeddingID=?",
                           (personID, row["embeddingID"]))

        # Commit the changes to the database
        conn.commit()
                 
    





def create_persons_table(cursor, conn):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Persons (
        personID INTEGER PRIMARY KEY AUTOINCREMENT,
        avgEmbedding BLOB,
        sampleEmbedding INTEGER,
        embeddingCount INTEGER
    )
    ''')
    conn.commit()




def extractFacesFromDb(conn, cursor):
    cursor.execute("SELECT * FROM FaceGroups")
    rows = cursor.fetchall()
    counter = 0
    for row in rows:
        imagePaths = json.loads(row[2])  # Corrected the variable name
        facialAreas = json.loads(row[3])  # Should be a list of dictionaries
        
        for imagePath, facialArea in zip(imagePaths, facialAreas):
            img = cv2.imread(imagePath)
            
            # Extract facial area details from the dictionary
            x = facialArea.get("x")
            y = facialArea.get("y")
            w = facialArea.get("w")
            h = facialArea.get("h")

            if img is not None:
                # Extract the face region
                face = img[y:y+h, x:x+w]
                
                # Save the face region. Modify the path to avoid overwriting the original image.
                fileName = os.path.basename(imagePath)
                face_image_path = os.path.join('test', fileName.replace(".jpg", "_face" + str(counter) + ".jpg"))  # Modify the file name
                counter+=1
                cv2.imwrite(face_image_path, face)
            else:
                print(f"Failed to read image at path: {imagePath}")

def createFaceEmbeddingsTable(conn, cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_embeddings (
        embeddingID INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding BLOB,
        mediaPath TEXT,
        facial_area TEXT,
        type TEXT,
        movieFrameNumber TEXT,
        personID INTEGER
    )
    ''')
                   
    conn.commit()

def extractAndSaveEmbeddingsFromImage(conn, cursor, image, media_path, mediaType, movieFrameNumber = None):
    # Get embeddings from DeepFace
    model_name = "Dlib"
    embeddings_data = DeepFace.represent(img_path=image, model_name=model_name, enforce_detection=False, detector_backend="retinaface")
    
    for data in embeddings_data:
        embedding = data['embedding']
        # Serialize the entire numpy array (including dtype) - to get the right format when reading them eg. float64 float32..
        serialized_embedding = pickle.dumps(embedding)
        facial_area = data['facial_area']
        cursor.execute("INSERT INTO face_embeddings (embedding, mediaPath, facial_area, type, movieFrameNumber) VALUES (?, ?, ?, ?, ?)",
                       serialized_embedding, media_path, json.dumps(facial_area), mediaType, movieFrameNumber)

    conn.commit()

def handleUnprocessedMediaItems(conn, cursor):
    # this method will 1.get unprocessed MediaItems
    # 2.extract and save face embeddings in faceEmbeddings table 
    # 3.cluster them into Persons Table

    # 1. get the unprocessed MediaItems
    facesUnprocessedMediaItems = get_unmapped_media_paths()
    if not facesUnprocessedMediaItems:
        print("No unprocessed media items found.")
        return
    # 2.iterate each media item and save it's embeddings
    for media_item in facesUnprocessedMediaItems:
        id, media_path, media_type = media_item
        print(f"Processing: {media_path} (Type: {media_type})")
        if media_type == "image":
            extractAndSaveEmbeddingsFromImage(conn, cursor, media_path, media_path, media_type)
        elif media_type == "movie":
            extracted_frames = extract_frames_from_movie(media_path)
            for frame, frameNumber in extracted_frames:
                extractAndSaveEmbeddingsFromImage(conn, cursor, frame, media_path, media_type, frameNumber)        
    conn.commit()  
    # 3. cluster embeddings










def main():
    # testEmbeddingsVsVerify()
    # test2FacesImage()
    # exit(0)
    
    
    conn = sqlite3.connect('faces.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()

    
    # cursor.execute('''
    # CREATE TABLE IF NOT EXISTS FaceGroups (
    #     personID INTEGER PRIMARY KEY,
    #     averageEmbedding BLOB,
    #     imagePaths TEXT,
    #     facialArea TEXT,
    #     embeddingCount INTEGER,
    #     uniqueImagePathCount INTEGER  -- New column to track unique image paths
    # )
    # ''')

    # conn.commit()
    createFaceEmbeddingsTable(conn, cursor)

    # extractFacesFromDb(conn, cursor)
    # conn.close()
    # exit(0)

    

    conn.close()
    print("Processing completed.")

if __name__ == "__main__":
    main()