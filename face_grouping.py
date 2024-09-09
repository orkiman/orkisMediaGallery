import base64
import io
import json
import os
import sqlite3
import cv2
from deepface import DeepFace
from deepface.modules import verification
import numpy as np
from PIL import Image

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
 
    
def extract_frames_from_video_slow(video_path, interval=100):
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

def extract_frames_from_video(video_path, interval=100):
    video = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while frame_count < 2000: # limit the number of images to 20
        # Set the video position to the desired frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        success, frame = video.read()
        if not success:
            break
        
        # Append a tuple containing both the frame and the frame number
        frames.append((frame, frame_count))
        
        # Move to the next frame of interest
        frame_count += interval
        print("Frame count: ", frame_count)
    
    video.release()
    return frames

def clearClustering(conn, cursor):
    cursor.execute("UPDATE faceEmbeddings SET personID = NULL")
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
    
    min_distance = float('inf')
    matchingEmbeddingTuple = None
    
    embedding_id_1, serialized_new_embedding = new_embedding_tuple

    new_embedding = np.frombuffer(serialized_new_embedding, dtype=np.float64)

    
    for recognized_embedding_tuple in recognized_embeddings_list:
        embedding_id_2, serialized_recognized_embedding, personID = recognized_embedding_tuple
        recognized_embedding = np.frombuffer(serialized_recognized_embedding, dtype=np.float64)
        # distance = verification.find_euclidean_distance(new_embedding, recognized_embedding)
        distance = verification.find_cosine_distance(new_embedding, recognized_embedding)
        if distance < min_distance and distance < thresholds["ArcFace"]['cosine']:
        # if distance < min_distance and distance < 0.06 :#thresholds[model_name]['cosine']:
            min_distance = distance
            matchingEmbeddingTuple = recognized_embedding_tuple
    
    if matchingEmbeddingTuple is not None:
        print("found. distance: ", min_distance)
    return matchingEmbeddingTuple 


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

def chinese_whispers_fixed_iterations_old(embeddings, threshold=0.5, iterations=20):

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

def chinese_whispers_max_iterations(embeddings, threshold=0.5, iterations=20, safety_zone=3):
    # Step 0: Normalize the embeddings
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
    # test results : embeddings are normlized good (close to 1)
    for embedding in embeddings:
        print (np.linalg.norm(embedding))


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

    unchanged_iterations = 0  # Counter for unchanged iterations
    prev_labels = None  # Variable to store previous labels

    # Step 4: Run the Chinese Whispers algorithm
    for iteration in range(iterations):
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

        # Collect current labels
        current_labels = [G.nodes[node]['label'] for node in G.nodes()]
        
        # Compare with previous labels to check if unchanged
        if prev_labels is not None and current_labels == prev_labels:
            unchanged_iterations += 1
        else:
            unchanged_iterations = 0  # Reset counter if labels changed

        prev_labels = current_labels.copy()  # Save current labels for next comparison
        
        print(f"Iteration {iteration + 1}: Unchanged for {unchanged_iterations} iterations")

        # Exit if unchanged for safety zone iterations
        if unchanged_iterations >= safety_zone:
            print(f"Stopping early at iteration {iteration + 1} after {unchanged_iterations} unchanged iterations.")
            break

    # Step 5: Extract clusters
    clusters = {}
    for node in G.nodes():
        label = G.nodes[node]['label']
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)

    return list(clusters.values())

import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics.pairwise import cosine_similarity

# def perform_clustering_not_clipped(embeddings, method='optics', **kwargs):
#     # Normalize embeddings
#     embeddings = np.array([embedding / np.linalg.norm(embedding) for embedding in embeddings])
    
#     # Compute cosine similarity matrix
#     similarity_matrix = cosine_similarity(embeddings)
    
#     if method == 'optics':
#         # OPTICS clustering
#         clusterer = OPTICS(metric='precomputed', **kwargs)
#         cluster_labels = clusterer.fit_predict(1 - similarity_matrix)  # Convert similarity to distance
#     elif method == 'dbscan':
#         # DBSCAN clustering
#         clusterer = DBSCAN(metric='precomputed', **kwargs)
#         cluster_labels = clusterer.fit_predict(1 - similarity_matrix)  # Convert similarity to distance
#     else:
#         raise ValueError("Invalid method. Choose 'optics' or 'dbscan'.")

#     # Group embeddings by cluster
#     clusters = {}
#     for i, label in enumerate(cluster_labels):
#         if label not in clusters:
#             clusters[label] = []
#         clusters[label].append(i)

#     # Remove noise points (label -1) if any
#     if -1 in clusters:
#         del clusters[-1]

#     return list(clusters.values())

def perform_clustering(embeddings, method='optics', **kwargs):
    # Convert list of embeddings to numpy array if it's not already
    embeddings = np.array(embeddings)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    # check print this should be close to 1
    for normalized_embedding in normalized_embeddings:
        print (np.linalg.norm(normalized_embedding))
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(normalized_embeddings)
    
    # Convert similarity to distance, ensuring non-negative values
    distance_matrix = np.clip(1 - similarity_matrix, 0, 2)
    
    if method == 'optics':
        # OPTICS clustering
        clusterer = OPTICS(metric='precomputed', **kwargs)
        cluster_labels = clusterer.fit_predict(distance_matrix)
    elif method == 'dbscan':
        # DBSCAN clustering
        clusterer = DBSCAN(metric='precomputed', **kwargs)
        cluster_labels = clusterer.fit_predict(distance_matrix)
    else:
        raise ValueError("Invalid method. Choose 'optics' or 'dbscan'.")

    # Group embeddings by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    # Remove noise points (label -1) if any
    if -1 in clusters:
        del clusters[-1]

    return list(clusters.values())
# Example usage:
# optics_clusters = perform_clustering(embeddings, method='optics', min_samples=5, xi=0.05)
# dbscan_clusters = perform_clustering(embeddings, method='dbscan', eps=0.5, min_samples=5)

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

# def prepareAndCluster(conn, cursor):
#     cursor.execute("SELECT embedding, embeddingID FROM faceEmbeddings")
#     faceEmbeddings_rows = cursor.fetchall()
#     embeddings = []
#     embeddingIDs = []
#     for faceEmbeddings_row in faceEmbeddings_rows:
#         # embeddingID = faceEmbeddings_row["embeddingID"]
#         # Deserialize the stored binary data back into a numpy array
#         serialized_face_embedding = faceEmbeddings_row["embedding"]
#         face_embedding = np.frombuffer(serialized_face_embedding, dtype=np.float64)
#         embeddings.append(face_embedding)
#         embeddingIDs.append(faceEmbeddings_row["embeddingID"])

    
#     for embedding in embeddings:
#         print(np.linalg.norm(embedding))
#     # clusters = chinese_whispers_max_iterations(embeddings, threshold=0.55)#0.41
#     # clusters = perform_clustering(embeddings, method='optics', min_samples=5, xi=0.01)
#     clusters = perform_clustering(embeddings, method='dbscan', eps=0.1, min_samples=2)

#     # total_items_in_clusters = sum(len(cluster) for cluster in clusters)
#     # print(f"Total items in all clusters: {total_items_in_clusters}")
    
#     print(f"Number of clusters: {len(clusters)}")
#     # first set unclustered faces
#     cursor.execute("UPDATE faceEmbeddings SET personID = -1")
#     personID = 0
#     for cluster in clusters:
#         print(f"Cluster {personID}: {cluster}")
#         if not cluster:
#             continue  # Skip empty clusters

#         # For single-item clusters
#         if len(cluster) == 1:
#             cluster_str = f"({embeddingIDs[cluster[0]]})"
#         else:
#             # Use list comprehension to convert cluster indices to embeddingIDs and join them as a string for the SQL query
#             cluster_str = str(tuple(embeddingIDs[i] for i in cluster))

#         # Debugging output to check the generated SQL query
#         # print(f"UPDATE faceEmbeddings SET personID = ? WHERE embeddingID IN {cluster_str}", (personID,))
        
#         # Execute the SQL UPDATE statement
#         cursor.execute(f"UPDATE faceEmbeddings SET personID = ? WHERE embeddingID IN {cluster_str}", (personID,))

#         personID += 1
        

#     conn.commit() 


def get_normalized_embeddings(cursor):
    cursor.execute("SELECT embedding, embeddingID FROM faceEmbeddings")
    faceEmbeddings_rows = cursor.fetchall()
    
    embeddings = []
    embeddingIDs = []
    
    for faceEmbeddings_row in faceEmbeddings_rows:
        # Deserialize the stored binary data back into a numpy array
        serialized_face_embedding = faceEmbeddings_row["embedding"]
        face_embedding = np.frombuffer(serialized_face_embedding, dtype=np.float64)
        
        # Normalize the embedding
        norm = np.linalg.norm(face_embedding)
        normalized_embedding = face_embedding / norm if norm != 0 else face_embedding
        
        embeddings.append(normalized_embedding)
        embeddingIDs.append(faceEmbeddings_row["embeddingID"])

    return embeddings, embeddingIDs


# def perform_clustering_and_update_db(conn, cursor, embeddings, embeddingIDs):
#     # Perform clustering on the embeddings
#     clusters = perform_clustering(embeddings, method='dbscan', eps=0.5, min_samples=2)

#     print(f"Number of clusters: {len(clusters)}")
    
#     # First set unclustered faces
#     cursor.execute("UPDATE faceEmbeddings SET personID = -1")
    
#     personID = 0
#     for cluster in clusters:
#         print(f"Cluster {personID}: {cluster}")
        
#         if not cluster:
#             continue  # Skip empty clusters

#         # For single-item clusters
#         if len(cluster) == 1:
#             cluster_str = f"({embeddingIDs[cluster[0]]})"
#         else:
#             cluster_str = str(tuple(embeddingIDs[i] for i in cluster))

#         # Execute the SQL UPDATE statement
#         cursor.execute(f"UPDATE faceEmbeddings SET personID = ? WHERE embeddingID IN {cluster_str}", (personID,))

#         personID += 1

#     conn.commit()

def update_db_personID_by_clusters(conn, cursor, clusters, embeddingIDs):
    
    # First set unclustered faces
    cursor.execute("UPDATE faceEmbeddings SET personID = -1")
    
    personID = 0
    for cluster in clusters:
        # print(f"Cluster {personID}: {cluster}")
        
        if not cluster:
            continue  # Skip empty clusters

        # For single-item clusters
        if len(cluster) == 1:
            cluster_str = f"({embeddingIDs[cluster[0]]})"
        else:
            cluster_str = str(tuple(embeddingIDs[i] for i in cluster))

        # Execute the SQL UPDATE statement
        cursor.execute(f"UPDATE faceEmbeddings SET personID = ? WHERE embeddingID IN {cluster_str}", (personID,))

        personID += 1

    conn.commit()

# ********save faces to db********

def createFaceImagesTable(conn, cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS faceImages (
    id SERIAL PRIMARY KEY,
    personID TEXT,       -- Corresponds to Face.PersonID
    faceImage TEXT,      -- Base64 string (corresponds to Face.ImageData)
    mediaCount TEXT      -- Number of media items (corresponds to Face.MediaCount)
    )
    ''')
                   
    conn.commit()


def save_faces_to_db(conn, cursor):
    # Create the faceImages table if it doesn't exist
    createFaceImagesTable(conn, cursor)
    
    # Clear the faceImages table
    cursor.execute("DELETE FROM faceImages")
    conn.commit()

    # SQL query to fetch data
    query = '''
    SELECT 
        fe.personID,
        m.absoluteFilePath,
        fe.facial_area,
        m.mediaType,
        fe.videoFrameNumber,
        COUNT(DISTINCT fe.mediaID) AS mediaCount
    FROM 
        faceEmbeddings fe
    JOIN 
        mediaItems m ON fe.mediaID = m.mediaID
    GROUP BY 
        fe.personID
    ORDER BY 
        mediaCount DESC;
    '''

    cursor.execute(query)
    rows = cursor.fetchall()
    total_faces = len(rows)

    # Iterate through each row with progress
    for idx, row in enumerate(rows):
        personID, absoluteFilePath, facialAreaJSON, mediaType, videoFrameNumber, mediaCount = row

        # Print progress
        print(f"preparing face sample {idx+1}/{total_faces}: PersonID {personID}, MediaType {mediaType}")

        # Fetch the image from disk or video frame
        if mediaType == "image":
            img = cv2.imread(absoluteFilePath)
            if img is None:
                print(f"Failed to read image from path {absoluteFilePath}")
                continue
        elif mediaType == "video":
            img = get_image_from_video_path(absoluteFilePath, videoFrameNumber)
            if img is None:
                print(f"Failed to get image from video at path {absoluteFilePath}, frame {videoFrameNumber}")
                continue

        # Get the face image (assuming `get_face_images_base64_from_image` is implemented)
        face_base64 = get_face_images_base64_from_image(img, facialAreaJSON)
        if face_base64 is None:
            print(f"Failed to get face image data for personID {personID}")
            continue

        # Convert base64 string to bytes to store as BLOB in the database
        face_image_blob = base64.b64decode(face_base64)

        # Insert the face image into the database
        insert_query = '''
        INSERT INTO faceImages (personID, faceImage, mediaCount)
        VALUES (?, ?, ?);
        '''
        cursor.execute(insert_query, (personID, face_image_blob, mediaCount))

    # Commit the transaction
    conn.commit()

    # Print final message
    print(f"Successfully processed and saved {total_faces} face images to the database.")


def get_face_images_base64_from_image(img, facial_area_json):
    # Parse the facial area JSON
    try:
        facial_area = json.loads(facial_area_json)
    except json.JSONDecodeError as e:
        print(f"Failed to parse facial areas: {e}")
        return None

    x, y, w, h = facial_area.get('x', 0), facial_area.get('y', 0), facial_area.get('w', 0), facial_area.get('h', 0)

    # Ensure the facial area is within bounds
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > img.shape[1]:
        w = img.shape[1] - x
    if y + h > img.shape[0]:
        h = img.shape[0] - y

    # Crop the image based on the facial area
    cropped_img = img[y:y+h, x:x+w]

    # Convert the cropped face to a PIL image for encoding
    cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

    # Encode the image to JPEG and then to Base64
    buffer = io.BytesIO()
    cropped_img_pil.save(buffer, format="JPEG")
    img_base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Return the face image in Base64 format
    return img_base64_str

def get_image_from_video_path(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None



from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

def analyze_embedding_distances(embeddings, sample_size=1000):
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    
    # If there are too many embeddings, sample a subset
    if len(normalized_embeddings) > sample_size:
        indices = np.random.choice(len(normalized_embeddings), sample_size, replace=False)
        sample_embeddings = normalized_embeddings[indices]
    else:
        sample_embeddings = normalized_embeddings
    
    # Compute pairwise distances
    distances = cosine_distances(sample_embeddings)
    
    # Plot histogram of distances
    plt.figure(figsize=(10, 6))
    plt.hist(distances.flatten(), bins=50, edgecolor='black')
    plt.title('Distribution of Cosine Distances between Embeddings')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Frequency')
    plt.show()
    
    # Print some statistics
    print(f"Min distance: {np.min(distances)}")
    print(f"Max distance: {np.max(distances)}")
    print(f"Mean distance: {np.mean(distances)}")
    print(f"Median distance: {np.median(distances)}")
    print(f"25th percentile: {np.percentile(distances, 25)}")
    print(f"75th percentile: {np.percentile(distances, 75)}")

# Usage:
# analyze_embedding_distances(embeddings)

def two_step_clustering(embeddings, eps1=0.05, min_samples1=2, eps2=0.2, min_samples2=2):
    # Ensure embeddings is a NumPy array
    embeddings = np.array(embeddings)
    
    # Step 1: Initial clustering
    initial_clusterer = DBSCAN(eps=eps1, min_samples=min_samples1, metric='cosine')
    initial_labels = initial_clusterer.fit_predict(embeddings)
    
    # Compute centroids of initial clusters
    centroids = []
    for label in set(initial_labels):
        if label != -1:  # Ignore noise points
            # Use np.where to find the indices of the embeddings in this cluster
            cluster_indices = np.where(initial_labels == label)[0]
            cluster_points = embeddings[cluster_indices]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
    
    # Step 2: Cluster the centroids
    if len(centroids) > 1:
        centroid_clusterer = DBSCAN(eps=eps2, min_samples=min_samples2, metric='cosine')
        centroid_labels = centroid_clusterer.fit_predict(centroids)
        
        # Map initial clusters to final clusters
        final_labels = np.zeros_like(initial_labels) - 1
        for i, label in enumerate(initial_labels):
            if label != -1:
                final_labels[i] = centroid_labels[label]
    else:
        final_labels = initial_labels
    
    # Group embeddings by final cluster
    clusters = {}
    for i, label in enumerate(final_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    # Remove noise points (label -1) if any
    if -1 in clusters:
        del clusters[-1]
    
    return list(clusters.values())


def cosine_distance(embedding1, embedding2):
    """Calculate cosine distance between two embeddings."""
    # cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    # return 1 - cosine_similarity
    return verification.find_cosine_distance(embedding1, embedding2)
    

# def is_similar_to_group(embedding, group, embeddings, threshold):
#     """Check if an embedding is similar to all members of the group."""
#     for member in group:
#         if cosine_distance(embedding, embeddings[member]) >= threshold:
#             return False  # If it's not similar to any one member, return False
#     return True  # Similar to all members

# def group_embeddings(embeddings, threshold=0.68):
#     """Group embeddings where every member of a group is similar to every other member."""
#     groups = []  # List of groups, each group is a list of indices of embeddings

#     for i, embedding in enumerate(embeddings):
#         added_to_group = False
        
#         for group in groups:
#             # Check if the embedding is similar to all members of the group
#             if is_similar_to_group(embedding, group, embeddings, threshold):
#                 group.append(i)  # Add to the group if similar to all members
#                 added_to_group = True
#                 # break  # No need to check other groups
        
#         if not added_to_group:
#             # If it didn't match any group, create a new group
#             groups.append([i])
    
#     return groups

from scipy.spatial.distance import cosine as cosine_distance
def compute_group_average(group, embeddings):
    """Compute the average (centroid) embedding of a group."""
    group_embeddings = np.array([embeddings[i] for i in group])
    return np.mean(group_embeddings, axis=0)

def is_similar_to_group_average(embedding, group_average, threshold):
    """Check if an embedding is similar to a given group average embedding."""
    if cosine_distance(embedding, group_average) < threshold:
        return True
    return False

def group_embeddings(embeddings, threshold=0.55):
    """Group embeddings based on similarity to the group average."""
    groups = []  # List of groups, each group is a list of indices of embeddings

    for i, embedding in enumerate(embeddings):
        added_to_group = False
        
        for group in groups:
            group_average = compute_group_average(group, embeddings)
            # Check if the embedding is similar to the average embedding of the group
            if is_similar_to_group_average(embedding, group_average, threshold):
                group.append(i)  # Add to the group if similar to the group average
                added_to_group = True
                break  # No need to check other groups if added
        
        if not added_to_group:
            # If it didn't match any group, create a new group
            groups.append([i])
    
    return groups

def merge_group_0_with_others(groups, embeddings, threshold=0.7):
    """Merge embeddings of group 0 into other groups based on similarity to their averages."""
    # Extract group 0 indices and other group indices
    group_0 = groups[0]
    other_groups = groups[1:]
    
    # Compute the averages of the other groups
    other_group_averages = [compute_group_average(group, embeddings) for group in other_groups]
    
    # Reassign each embedding from group 0 to the most similar other group
    for idx in group_0:
        embedding = embeddings[idx]
        added_to_group = False
        
        # Find the most similar group based on the average embedding
        for group, group_average in zip(other_groups, other_group_averages):
            if is_similar_to_group_average(embedding, group_average, threshold):
                group.append(idx)  # Add embedding to the most similar group
                added_to_group = True
                break
        
        # If it doesn't fit any existing group, create a new group (optional)
        if not added_to_group:
            other_groups.append([idx])
    
    return other_groups

def main():
    
    conn = sqlite3.connect('orkisMediaGallery.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()

    createFaceEmbeddingsTable(conn, cursor)
    clearClustering(conn, cursor)
    handleUnprocessedMediaItems(conn, cursor)
    # prepareAndCluster(conn, cursor)
    
    # Get normalized embeddings
    embeddings, embeddingIDs = get_normalized_embeddings(cursor)
    # for embedding in embeddings:
    #     norm = np.linalg.norm(embedding)
    #     print(f"Norm of embedding: {norm}")

    # Analyze distances
    analyze_embedding_distances(embeddings)

    # Perform clustering and update the database
    # perform_clustering_and_update_db(conn, cursor, embeddings, embeddingIDs)
    # Perform clustering on the embeddings
    # clusters = perform_clustering(embeddings, method='dbscan', eps=0.5, min_samples=2)
    # clusters = two_step_clustering(embeddings)#, eps1=0.3, min_samples1=2, eps2=0.5, min_samples2=2)
    # print (clusters)
    clusters1 = group_embeddings(embeddings)
    clusters = merge_group_0_with_others(clusters1, embeddings)
    # print (clusters)

    update_db_personID_by_clusters(conn, cursor, clusters, embeddingIDs)

    save_faces_to_db(conn, cursor)

    conn.close()

if __name__ == "__main__":
    main()

