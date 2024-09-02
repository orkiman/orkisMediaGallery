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
 
    
def extract_frames_from_video(video_path, interval=100):
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

def clearClustering(conn, cursor):
    cursor.execute("UPDATE faceEmbeddings SET personID = NULL")
    cursor.execute("DELETE FROM Persons")
    conn.commit()
    print ("clustering cleared")

def clusterEmbeddings(conn , cursor):
    print("clustering started")
    cursor.execute("SELECT * FROM faceEmbeddings WHERE PERSONID IS NULL")
    # cursor.execute("SELECT * FROM faceEmbeddings WHERE PERSONID IS NULL ORDER BY embeddingID DESC")
    faceEmbeddings_rows = cursor.fetchall()
    model_name = "Dlib"
    # iterating unpamed face embeddings from faceEmbeddings table
    for faceEmbeddings_row in faceEmbeddings_rows:
        embeddingID = faceEmbeddings_row["embeddingID"]
        # Deserialize the stored binary data back into a numpy array
        serialized_face_embedding = faceEmbeddings_row["embedding"]
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
            threshold = 0.069
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
                
                # Update the personID in the faceEmbeddings table
                cursor.execute("UPDATE faceEmbeddings SET personID=? WHERE embeddingID=?",
                               (personID, embeddingID))
            else:
                #Insert as a new person
                cursor.execute("INSERT INTO Persons (avgEmbedding, embeddingCount, faceSampleEmbeddingID) VALUES (?, ?, ?)", 
                               (np.array(face_embedding, dtype=np.float64).tobytes(), 1, embeddingID))
                
                # Insert the new personID in the faceEmbeddings table
                personID = cursor.lastrowid
                cursor.execute("UPDATE faceEmbeddings SET personID=? WHERE embeddingID=?",
                               (personID, embeddingID))
        else:
            #Insert as the first person
            cursor.execute("INSERT INTO Persons (avgEmbedding, embeddingCount, faceSampleEmbeddingID) VALUES (?, ?, ?)", 
                           (np.array(face_embedding, dtype=np.float64).tobytes(), 1, embeddingID))
            
            # Insert the new personID in the faceEmbeddings table
            personID = cursor.lastrowid
            cursor.execute("UPDATE faceEmbeddings SET personID=? WHERE embeddingID=?",
                           (personID, embeddingID))

        # Commit the changes to the database
        conn.commit()
    print("clustering done")







def cluster_embeddings_without_average(conn, cursor):
    # iterate new embeddings
    # for each one, pass recognized embeddings list to get_closest_embedding
    # if found, update personID    
    # if not found, insert as new person
    # anyway add them to recognized embedding list
    
    print("Clustering started")
    cursor.execute("SELECT embeddingID, embedding FROM faceEmbeddings WHERE personID IS NULL")
    new_embeddings_list = cursor.fetchall()  
    cursor.execute("SELECT embeddingID, embedding, personID FROM faceEmbeddings WHERE personID IS NOT NULL")
    recognized_embeddings_list = cursor.fetchall()

    # iterating unnamed face embeddings from faceEmbeddings table
    for new_embedding_tuple in new_embeddings_list:
        
        matching_embedding_tuple = get_closest_embedding(new_embedding_tuple, recognized_embeddings_list)
        if matching_embedding_tuple is not None:
            # found existing person
            # update db
            personID = matching_embedding_tuple[2]
            
        else:
            # insert new person
            cursor.execute("INSERT INTO PERSONS (faceSampleEmbeddingID) VALUES (?)",
                           (new_embedding_tuple[0],))
            personID = cursor.lastrowid            
        cursor.execute("UPDATE faceEmbeddings SET personID=? WHERE embeddingID=?",
                           (personID, new_embedding_tuple[0]))
        # create a new tuple with personID and add it to the recognized list
        new_embedding_tuple = (new_embedding_tuple[0], new_embedding_tuple[1], personID)
        recognized_embeddings_list.append(new_embedding_tuple)
    
    # Commit the transaction
    conn.commit()


            

    
                 
def get_closest_embedding(new_embedding_tuple, recognized_embeddings_list):
    # this func gets new embedding and list of recognized embeddings
    # returns the closest embedding in the list from the recognized embedding
    # as tuple, if not found, returns None
    model_name = "Dlib"
    min_distance = float('inf')
    matchingEmbeddingTuple = None
    
    embedding_id_1, serialized_new_embedding = new_embedding_tuple

    new_embedding = np.frombuffer(serialized_new_embedding, dtype=np.float64)

    
    for recognized_embedding_tuple in recognized_embeddings_list:
        embedding_id_2, serialized_recognized_embedding, personID = recognized_embedding_tuple
        recognized_embedding = np.frombuffer(serialized_recognized_embedding, dtype=np.float64)
        # distance = verification.find_euclidean_distance(new_embedding, recognized_embedding)
        distance = verification.find_cosine_distance(new_embedding, recognized_embedding)
        if distance < min_distance and distance < thresholds[model_name]['cosine']:
        # if distance < min_distance and distance < 0.06 :#thresholds[model_name]['cosine']:
            min_distance = distance
            matchingEmbeddingTuple = recognized_embedding_tuple
    
    if matchingEmbeddingTuple is not None:
        print("found. distance: ", min_distance)
    return matchingEmbeddingTuple 

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
    CREATE TABLE IF NOT EXISTS faceEmbeddings (
        embeddingID INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding BLOB,
        mediaID INTEGER,
        facial_area TEXT,
        videoFrameNumber INTEGER,
        personID INTEGER
    )
    ''')
                   
    conn.commit()

def extractAndSaveEmbeddingsFromImage(conn, cursor, imageOrPath, media_ID, video_frame_number = 0):
    
    # Get embeddings from DeepFace
    # model_name = "Dlib"
    model_name = "ArcFace"
    try:
        embeddings_data = DeepFace.represent(img_path=imageOrPath, model_name=model_name, enforce_detection=True, detector_backend="retinaface")
    except Exception as e:
        print("no face detected or other error : ", e)
        # no face detected
        return
    
    for data in embeddings_data:
        embedding = data['embedding']
        # Serialize the full numpy array (forcing 64-bit precision)
        embedding_array = np.array(embedding, dtype=np.float64)
        serialized_embedding = embedding_array.tobytes()

        facial_area = data['facial_area']
        cursor.execute("INSERT INTO faceEmbeddings (embedding, mediaID, facial_area, videoFrameNumber) VALUES (?, ?, ?, ?)",
                       (serialized_embedding, media_ID, json.dumps(facial_area),  video_frame_number))



def handleUnprocessedMediaItems(conn, cursor):
    print("handleUnprocessedMediaItems started", flush=True)
    # print("testing printing and returning")
    # return
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
        # 2. Iterate each media item and save its embeddings save every X items
        processed_items = []
        chunkSize = 10
        itemsLeft = len(facesUnprocessedMediaItems)
        for media_item_info in facesUnprocessedMediaItems:
            mediaID, media_path, media_type = media_item_info 
            print(f"Processing: {media_path} (Type: {media_type})", flush=True)
            if media_type == "image":
                extractAndSaveEmbeddingsFromImage(conn, cursor, media_path, mediaID)
            elif media_type == "video":
                extracted_frames = extract_frames_from_video(media_path)
                for frame, frameNumber in extracted_frames:
                    extractAndSaveEmbeddingsFromImage(conn, cursor, frame, mediaID, frameNumber)
            processed_items.append(mediaID)
            if len(processed_items) % chunkSize == 0:
                cursor.execute("DELETE FROM facesUnprocessedMediaItems WHERE mediaID IN " + str(tuple(processed_items)))
                conn.commit()
                processed_items = []
                itemsLeft = itemsLeft - chunkSize
                print (f"items left for processing: {itemsLeft}", flush=True)

        # print(f"prcessed {len(facesUnprocessedMediaItems)} items")

        # Clear unprocessed media items
        # disable for testing
        cursor.execute("DELETE FROM facesUnprocessedMediaItems")
        
        #  Commit after all operations are successful
        conn.commit()
        
        # # 3. Cluster embeddings after committing
        # clusterEmbeddings(conn, cursor)

    except Exception as e:
        # Roll back in case of any failure
        conn.rollback()
        print(f"An error occurred: {e}")
        raise  # Optionally re-raise the error if you want it to propagate



import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def chinese_whispers_fix_iterations(embeddings, threshold=0.5, iterations=20):

    # Step 0: Normalize the embeddings
    # normalized_embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
    
    # Step 1: Construct the graph
    G = nx.Graph()

    for i, embedding in enumerate(embeddings):
        G.add_node(i)

    # Step 2: Add edges based on similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    # Step 3: Initialize labels
    for node in G.nodes():
        G.nodes[node]['label'] = node

    # Step 4: Run the Chinese Whispers algorithm
    for _ in range(iterations):
        nodes = list(G.nodes())
        np.random.shuffle(nodes)
        
        for node in nodes:
            neighbors = G[node]
            if not neighbors:
                continue
            
            # Get the labels of the neighbors
            labels = [G.nodes[neighbor]['label'] for neighbor in neighbors]
            # Assign the most frequent label to the current node
            most_frequent_label = max(set(labels), key=labels.count)
            G.nodes[node]['label'] = most_frequent_label

    # Step 5: Extract clusters
    clusters = {}
    for node in G.nodes():
        label = G.nodes[node]['label']
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)

    return list(clusters.values())

def chinese_whispers(embeddings, threshold=0.5, max_iterations=100, stability_check=15):

    # Step 0: Normalize the embeddings
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
    
    # Step 1: Construct the graph
    G = nx.Graph()

    for i, embedding in enumerate(embeddings):
        G.add_node(i)

    # Step 2: Add edges based on similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    # Step 3: Initialize labels
    for node in G.nodes():
        G.nodes[node]['label'] = node

    # Step 4: Run the Chinese Whispers algorithm with stability check
    stable_iterations = 0
    for iteration in range(max_iterations):
        nodes = list(G.nodes())
        np.random.shuffle(nodes)
        
        changes = 0
        for node in nodes:
            neighbors = G[node]
            if not neighbors:
                continue
            
            # Get the labels of the neighbors
            labels = [G.nodes[neighbor]['label'] for neighbor in neighbors]
            # Assign the most frequent label to the current node
            most_frequent_label = max(set(labels), key=labels.count)
            
            if G.nodes[node]['label'] != most_frequent_label:
                G.nodes[node]['label'] = most_frequent_label
                changes += 1
        
        # If no changes occurred, count towards stability
        if changes == 0:
            stable_iterations += 1
        else:
            stable_iterations = 0  # Reset stability count if there were changes

        # Stop if the graph has been stable for the required number of iterations
        if stable_iterations >= stability_check:
            print(f"Converged after {iteration+1} iterations with stability check.")
            break

    # Step 5: Extract clusters
    clusters = {}
    for node in G.nodes():
        label = G.nodes[node]['label']
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)

    return list(clusters.values())

# Usage example
# embeddings = np.random.rand(100, 512)  # Replace with your actual embeddings
# clusters = chinese_whispers(embeddings, threshold=0.7, iterations=20)

# print(f"Number of clusters: {len(clusters)}")

def prepareAndClusterChineseWhispers(conn, cursor):
    cursor.execute("SELECT * FROM faceEmbeddings")
    faceEmbeddings_rows = cursor.fetchall()
    embeddings = []
    for faceEmbeddings_row in faceEmbeddings_rows:
        # embeddingID = faceEmbeddings_row["embeddingID"]
        # Deserialize the stored binary data back into a numpy array
        serialized_face_embedding = faceEmbeddings_row["embedding"]
        face_embedding = np.frombuffer(serialized_face_embedding, dtype=np.float64)
        embeddings.append(face_embedding)

    # clusters = chinese_whispers(embeddings, threshold=0.41, max_iterations=100)
    clusters = chinese_whispers_fix_iterations(embeddings, threshold=0.41, iterations=50)
    print(f"Number of clusters: {len(clusters)}")
    personID = 0
    for cluster in clusters:
        print(f"Cluster {personID}: {cluster}")
        if not cluster:
            continue  # Skip empty clusters

        # # updateDatabase
        # cursor.execute("INSERT INTO PERSONS (faceSampleEmbeddingID) VALUES (?)", (faceEmbeddings_rows[0]["embeddingID"],))                        
        # personID = cursor.lastrowid 
        
        # Convert the cluster list to a tuple
        if len(cluster) == 1:
            cluster_str = f"({cluster[0]})"  # Handle single-item clusters correctly
        else:
            cluster_str = str(tuple(cluster))    
          
        # Debugging output to check the generated SQL query
        # print(f"UPDATE faceEmbeddings SET personID = ? WHERE embeddingID IN {cluster_str}", (personID,))
        
        # Execute the SQL UPDATE statement
        cursor.execute(f"UPDATE faceEmbeddings SET personID = ? WHERE embeddingID -1 IN {cluster_str}", (personID,))

        personID += 1
        

    conn.commit() 










def main():
    
    conn = sqlite3.connect('orkisMediaGallery.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()

    createFaceEmbeddingsTable(conn, cursor)
    create_persons_table(conn, cursor)
    clearClustering(conn, cursor)
    handleUnprocessedMediaItems(conn, cursor)
    # clusterEmbeddings(conn, cursor)
    # cluster_embeddings_without_average(conn, cursor)
    prepareAndClusterChineseWhispers(conn, cursor)
    conn.close()

if __name__ == "__main__":
    main()