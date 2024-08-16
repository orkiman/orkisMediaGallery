import json
import os
import sqlite3
import cv2
from deepface import DeepFace
from deepface.modules import verification
import numpy as np

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
            frames.append(frame)
        
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

    

        



def process_new_image_old_using32float(cursor, image, image_path, conn):
    
    embeddings_data = DeepFace.represent(img_path=image, model_name="ArcFace", enforce_detection=False)
    for data in embeddings_data:
        embedding = np.array(data['embedding'], dtype=np.float32)
        facial_area = data['facial_area']
        
        cursor.execute("SELECT personID, averageEmbedding, embeddingCount FROM FaceGroups")
        rows = cursor.fetchall()
        print(f"Number of rows fetched: {len(rows)}")

        distances = []
        for row in rows:
            stored_embedding = np.frombuffer(row[1], dtype=np.float32)
            distance = verification.find_cosine_distance(stored_embedding, embedding)
            distances.append(distance)
        
        if distances:
            min_distance_idx = np.argmin(distances)
            if distances[min_distance_idx] < 0.68:  # Adjust threshold as needed 0.68
                # Update existing person
                personID = rows[min_distance_idx][0]
                current_avg_embedding = np.frombuffer(rows[min_distance_idx][1], dtype=np.float32)
                num_embeddings = rows[min_distance_idx][2]
                
                new_average_embedding = (current_avg_embedding * num_embeddings + embedding) / (num_embeddings + 1)
                
                cursor.execute("SELECT imagePaths, facialArea FROM FaceGroups WHERE personID=?", (personID,))
                image_paths, facial_areas = cursor.fetchone()
                image_paths = json.loads(image_paths)
                image_paths.append(image_path)
                facial_areas = json.loads(facial_areas)

                # # Ensure that facial_areas is a list
                # if not isinstance(facial_areas, list):
                #     facial_areas = [facial_areas]  # Convert to list if it's not already

                facial_areas.append(facial_area)
                # image_paths = json.loads(cursor.fetchone()[0])
                # if image_path:
                #     image_paths.append(image_path)

                # facial_areas = json.loads(cursor.fetchone()[1])
                # facial_areas.append(facial_area)
                
                cursor.execute("UPDATE FaceGroups SET averageEmbedding=?, imagePaths=?, facialArea=?, embeddingCount=? WHERE personID=?", 
                               (new_average_embedding.tobytes(), json.dumps(image_paths), json.dumps(facial_areas), num_embeddings + 1, personID))                
            else:
                # Insert as a new person
                cursor.execute("INSERT INTO FaceGroups (averageEmbedding, imagePaths, facialArea, embeddingCount) VALUES (?, ?, ?, ?)", 
                               (embedding.tobytes(), json.dumps([image_path] if image_path else []), json.dumps([facial_area]), 1))
        else:
            # Insert as the first person
            cursor.execute("INSERT INTO FaceGroups (averageEmbedding, imagePaths, facialArea, embeddingCount) VALUES (?, ?, ?, ?)", 
                           (embedding.tobytes(), json.dumps([image_path] if image_path else []), json.dumps([facial_area]), 1))
        conn.commit()


def process_new_image(cursor, image, image_path, conn):
    # Get embeddings from DeepFace
    embeddings_data = DeepFace.represent(img_path=image, model_name="ArcFace", enforce_detection=False)
    
    for data in embeddings_data:
        # Directly use the embedding as it is
        embedding = data['embedding']

        facial_area = data['facial_area']
        
        # Fetch all existing face groups from the database
        cursor.execute("SELECT personID, averageEmbedding, embeddingCount FROM FaceGroups")
        rows = cursor.fetchall()
        print(f"Number of rows fetched: {len(rows)}")

        distances = []
        for row in rows:
            # Convert the stored embedding back from bytes to a numpy array (float64 by default)
            stored_embedding = np.frombuffer(row[1], dtype=np.float64)
            # Calculate the distance between the current embedding and the stored embeddings
            distance = verification.find_cosine_distance(stored_embedding, embedding)
            distances.append(distance)
        
        if distances:
            min_distance_idx = np.argmin(distances)
            if distances[min_distance_idx] < 0.68:  # Adjust threshold as needed
                # Update existing person
                personID = rows[min_distance_idx][0]
                current_avg_embedding = np.frombuffer(rows[min_distance_idx][1], dtype=np.float64)
                num_embeddings = rows[min_distance_idx][2]
                
                # Calculate the new average embedding
                new_average_embedding = (current_avg_embedding * num_embeddings + embedding) / (num_embeddings + 1)
                
                # Retrieve and update the image paths and facial areas
                cursor.execute("SELECT imagePaths, facialArea FROM FaceGroups WHERE personID=?", (personID,))
                image_paths, facial_areas = cursor.fetchone()
                image_paths = json.loads(image_paths)
                image_paths.append(image_path)
                facial_areas = json.loads(facial_areas)
                facial_areas.append(facial_area)
                
                # Update the database with the new information
                cursor.execute("UPDATE FaceGroups SET averageEmbedding=?, imagePaths=?, facialArea=?, embeddingCount=? WHERE personID=?", 
                               (new_average_embedding.tobytes(), json.dumps(image_paths), json.dumps(facial_areas), num_embeddings + 1, personID))
            else:
                # Insert as a new person
                cursor.execute("INSERT INTO FaceGroups (averageEmbedding, imagePaths, facialArea, embeddingCount) VALUES (?, ?, ?, ?)", 
                               (np.array(embedding, dtype=np.float64).tobytes(), json.dumps([image_path] if image_path else []), json.dumps([facial_area]), 1))
        else:
            # Insert as the first person
            cursor.execute("INSERT INTO FaceGroups (averageEmbedding, imagePaths, facialArea, embeddingCount) VALUES (?, ?, ?, ?)", 
                           (np.array(embedding, dtype=np.float64).tobytes(), json.dumps([image_path] if image_path else []), json.dumps([facial_area]), 1))
        
        # Commit the changes to the database
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

def main():
    # testEmbeddingsVsVerify()
    # test2FacesImage()
    # exit(0)
    facesUnprocessedMediaItems = get_unmapped_media_paths()
    if not facesUnprocessedMediaItems:
        print("No unprocessed media items found.")
        return

    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()

    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS FaceGroups (
        personID INTEGER PRIMARY KEY,
        averageEmbedding BLOB,
        imagePaths TEXT,
        facialArea TEXT,
        embeddingCount INTEGER
    )
    ''')
    conn.commit()

    # extractFacesFromDb(conn, cursor)
    # conn.close()
    # exit(0)

    for media_item in facesUnprocessedMediaItems:
        id, media_path, media_type = media_item
        print(f"Processing: {media_path} (Type: {media_type})")
        if media_type == "image":
            process_new_image(cursor, media_path, media_path, conn)
        elif media_type == "movie":
            extracted_frames = extract_frames_from_movie(media_path)
            for frame in extracted_frames:
                process_new_image(cursor, frame, media_path, conn)
        
        conn.commit()  # Commit after processing each media item

    conn.close()
    print("Processing completed.")

if __name__ == "__main__":
    main()