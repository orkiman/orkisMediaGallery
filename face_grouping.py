import json
import os
import sqlite3
import cv2
from deepface import DeepFace
from deepface.modules import verification
import numpy as np

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

def get_unmapped_media_paths(conn, cursor):
    
    
    cursor.execute('''
                   SELECT name FROM sqlite_master WHERE type='table' AND name='facesUnprocessedMediaItems'
                   ''')
    table_exists = cursor.fetchone() is not None
    
    unmapped_media = []
    if table_exists:
        cursor.execute('SELECT * FROM facesUnprocessedMediaItems')
        unmapped_media = cursor.fetchall()  # Fetch both the file path and its type (image/video)
    
    conn.commit()
    return unmapped_media
 
    
def extract_frames_from_video(video_path, interval=20):
    video = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break
        
        if frame_count % interval == 0:
            # Append a tuple containing both the frame and the frame number
            frames.append((frame, frame_count))
        
        frame_count += 1
    
    video.release()
    return frames


def clusterEmbeddings(conn , cursor):
    cursor.execute("SELECT * FROM face_embeddings WHERE PERSONID IS NULL")
    face_embeddings_rows = cursor.fetchall()
    model_name = "Dlib"
    # iterating unpamed face embeddings from face_embeddings table
    for face_embeddings_row in face_embeddings_rows:
        embeddingID = face_embeddings_row["embeddingID"]
        # Deserialize the stored binary data back into a numpy array
        serialized_face_embedding = face_embeddings_row["embedding"]
        face_embedding = np.frombuffer(serialized_face_embedding, dtype=np.float64)

        cursor.execute("SELECT personID, avgEmbedding, embeddingCount from Persons")
        personsID_rows = cursor.fetchall()
        distances = []
        # iterating persons table
        for personID_row in personsID_rows:
            serialized_person_avg_embedding = personID_row["avgEmbedding"] 
            person_avg_embedding = np.frombuffer(serialized_person_avg_embedding, dtype=np.float64)
            personID = personID_row["personID"]
            embeddingCount = personID_row["embeddingCount"]
            distance = verification.find_cosine_distance(person_avg_embedding, face_embedding)
            distances.append((personID, distance, embeddingCount))
        
        if distances:
            min_distance_idx = min(enumerate(distances), key=lambda x: x[1][1])[0]
            threshold = thresholds[model_name]['cosine']
            if distances[min_distance_idx][1] < threshold :  # Adjust threshold as needed
                #update person table
                personID = distances[min_distance_idx][0]
                serialized_match_person_avg_embedding = personsID_rows[min_distance_idx][1] 
                match_person_avg_embedding = np.frombuffer(serialized_match_person_avg_embedding, dtype=np.float64)
                oldEmbeddingsCount = personsID_rows[min_distance_idx][2]
                newEmbeddingsCount = oldEmbeddingsCount + 1
                newAverageEmbedding = (match_person_avg_embedding * oldEmbeddingsCount + face_embedding) / (newEmbeddingsCount)                
                serialized_new_average_embedding = newAverageEmbedding.tobytes()

                cursor.execute("UPDATE Persons SET avgEmbedding=?, embeddingCount=? WHERE personID=?", 
                               (serialized_new_average_embedding, newEmbeddingsCount, personID))
                
                # Update the personID in the face_embeddings table
                cursor.execute("UPDATE face_embeddings SET personID=? WHERE embeddingID=?",
                               (personID, embeddingID))
            else:
                #Insert as a new person
                cursor.execute("INSERT INTO Persons (avgEmbedding, embeddingCount, faceSampleEmbeddingID) VALUES (?, ?, ?)", 
                               (np.array(face_embedding, dtype=np.float64).tobytes(), 1, embeddingID))
                
                # Insert the new personID in the face_embeddings table
                personID = cursor.lastrowid
                cursor.execute("UPDATE face_embeddings SET personID=? WHERE embeddingID=?",
                               (personID, embeddingID))
        else:
            #Insert as the first person
            cursor.execute("INSERT INTO Persons (avgEmbedding, embeddingCount, faceSampleEmbeddingID) VALUES (?, ?, ?)", 
                           (np.array(face_embedding, dtype=np.float64).tobytes(), 1, embeddingID))
            
            # Insert the new personID in the face_embeddings table
            personID = cursor.lastrowid
            cursor.execute("UPDATE face_embeddings SET personID=? WHERE embeddingID=?",
                           (personID, embeddingID))

        # Commit the changes to the database
        conn.commit()
                 
   

def create_persons_table(conn, cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Persons (
        personID INTEGER PRIMARY KEY AUTOINCREMENT,
        avgEmbedding BLOB,
        faceSampleEmbeddingID INTEGER,
        embeddingCount INTEGER
    )
    ''')
    conn.commit()



def createFaceEmbeddingsTable(conn, cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_embeddings (
        embeddingID INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding BLOB,
        mediaID INTEGER,
        facial_area TEXT,
        videoFrameNumber TEXT,
        personID INTEGER
    )
    ''')
                   
    conn.commit()

def extractAndSaveEmbeddingsFromImage(conn, cursor, imageOrPath, media_ID, video_frame_number = None):
    
    # Get embeddings from DeepFace
    model_name = "Dlib"
    embeddings_data = DeepFace.represent(img_path=imageOrPath, model_name=model_name, enforce_detection=False, detector_backend="retinaface")
    
    for data in embeddings_data:
        embedding = data['embedding']
        # Serialize the full numpy array (forcing 64-bit precision)
        embedding_array = np.array(embedding, dtype=np.float64)
        serialized_embedding = embedding_array.tobytes()

        facial_area = data['facial_area']
        cursor.execute("INSERT INTO face_embeddings (embedding, mediaID, facial_area, videoFrameNumber) VALUES (?, ?, ?, ?)",
                       (serialized_embedding, media_ID, json.dumps(facial_area),  video_frame_number))


# def old_handleUnprocessedMediaItems(conn, cursor):
#     # this method will 
#     # 1.get unprocessed MediaItems
#     # 2.extract and save face embeddings in faceEmbeddings table 
#     # 3.cluster them into Persons Table

#     try:
#         # 1. Get the unprocessed MediaItems
#         facesUnprocessedMediaItems = get_unmapped_media_paths(conn, cursor)
#         if not facesUnprocessedMediaItems:
#             print("No unprocessed media items found.")
#             return
        
#         # 2. Iterate each media item and save its embeddings
#         for media_item in facesUnprocessedMediaItems:
#             id, media_path, media_type = media_item
#             print(f"Processing: {media_path} (Type: {media_type})")
#             if media_type == "image":
                
#                 extractAndSaveEmbeddingsFromImage(conn, cursor, media_path, media_path, media_type)
#             elif media_type == "video":
#                 extracted_frames = extract_frames_from_video(media_path)
#                 for frame, frameNumber in extracted_frames:
#                     extractAndSaveEmbeddingsFromImage(conn, cursor, frame, media_path, media_type, frameNumber)
#         print(f"prcessed {len(facesUnprocessedMediaItems)} items")
        
#         # Clear unprocessed media items

#         # disable for testing
#         cursor.execute("DELETE FROM facesUnprocessedMediaItems")
        
#         #  Commit after all operations are successful
#         conn.commit()

#         # 3. Cluster embeddings after committing
#         clusterEmbeddings(conn, cursor)

#     except Exception as e:
#         # Roll back in case of any failure
#         conn.rollback()
#         print(f"An error occurred: {e}")
#         raise  # Optionally re-raise the error if you want it to propagate


def handleUnprocessedMediaItems(conn, cursor):
    # this method will 
    # 1.get unprocessed MediaItems
    # 2.extract and save face embeddings in faceEmbeddings table 
    # 3.cluster them into Persons Table
    try:
        # 1. Get the unprocessed MediaItems
        # quary = "SELECT mediaID, absoluteMediaPath, mediaType From mediaItems, facesUnprocessedMediaItems WHERE mediaItems.mediaID = facesUnprocessedMediaItems.mediaID"
        quary = '''
            SELECT 
            mediaItems.mediaID, 
            absoluteFilePath, 
            mediaType 
            FROM 
            mediaItems 
            INNER JOIN facesUnprocessedMediaItems 
            ON mediaItems.mediaID = facesUnprocessedMediaItems.mediaID
        '''
        cursor.execute(quary)
        facesUnprocessedMediaItems = cursor.fetchall()
        # 2. Iterate each media item and save its embeddings
        for media_item_info in facesUnprocessedMediaItems:
            mediaID, media_path, media_type = media_item_info 
            print(f"Processing: {media_path} (Type: {media_type})")
            if media_type == "image":
                extractAndSaveEmbeddingsFromImage(conn, cursor, media_path, mediaID)
            elif media_type == "video":
                extracted_frames = extract_frames_from_video(media_path)
                for frame, frameNumber in extracted_frames:
                    extractAndSaveEmbeddingsFromImage(conn, cursor, frame, mediaID, frameNumber)
        print(f"prcessed {len(facesUnprocessedMediaItems)} items")

        # Clear unprocessed media items
        # disable for testing
        cursor.execute("DELETE FROM facesUnprocessedMediaItems")
        
        #  Commit after all operations are successful
        conn.commit()

        # 3. Cluster embeddings after committing
        clusterEmbeddings(conn, cursor)

    except Exception as e:
        # Roll back in case of any failure
        conn.rollback()
        print(f"An error occurred: {e}")
        raise  # Optionally re-raise the error if you want it to propagate











def main():
    
    conn = sqlite3.connect('orkisMediaGallery.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()

    createFaceEmbeddingsTable(conn, cursor)
    create_persons_table(conn, cursor)
    handleUnprocessedMediaItems(conn, cursor)

    conn.close()

if __name__ == "__main__":
    main()