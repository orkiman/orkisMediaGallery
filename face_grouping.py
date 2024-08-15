import json
import os
import sqlite3
import cv2
from deepface import DeepFace
# from deepface.commons import distance as dst
from deepface.modules import verification

import numpy as np

baseDir = '/home/orkiman/Pictures/myPhotosTest'

def get_unmapped_media_paths(base_dir):
    # conn = sqlite3.connect(base_dir + '/faces.db')
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()
    
    # Check if the facesUnprocessedMediaItems table exists
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
    """Extract frames from a video at specified intervals and return them as NumPy arrays."""
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

def process_new_image(image, image_path=None):
    embeddings_data = DeepFace.represent(img_path=image, model_name="VGG-Face", enforce_detection=False)
    for data in embeddings_data:
        embedding = data['embedding']
        facial_area = data['facial_area']
        
        cursor.execute("SELECT personID, averageEmbedding, embeddingCount FROM FaceGroups")
        rows = cursor.fetchall()
        
        distances = []
        for row in rows:
            stored_embedding = np.frombuffer(row[1], dtype=np.float32)
            # distance = dst.findCosineDistance(stored_embedding, embedding)
            distance = verification.find_cosine_distance(stored_embedding, embedding)
            distances.append(distance)
        
        if distances:
            min_distance_idx = np.argmin(distances)
            if distances[min_distance_idx] < 0.4:  # Adjust the threshold as needed
                # Update existing person
                personID = rows[min_distance_idx][0]
                current_avg_embedding = np.frombuffer(rows[min_distance_idx][1], dtype=np.float32)
                num_embeddings = rows[min_distance_idx][2]
                
                # Calculate new average embedding
                new_average_embedding = (current_avg_embedding * num_embeddings + embedding) / (num_embeddings + 1)
                
                cursor.execute("SELECT imagePaths FROM FaceGroups WHERE personID=?", (personID,))
                image_paths = json.loads(cursor.fetchone()[0])
                if image_path:
                    image_paths.append(image_path)
                
                cursor.execute("UPDATE FaceGroups SET averageEmbedding=?, imagePaths=?, embeddingCount=? WHERE personID=?", 
                               (new_average_embedding.tobytes(), json.dumps(image_paths), num_embeddings + 1, personID))
            else:
                # Insert as a new person
                cursor.execute("INSERT INTO FaceGroups (averageEmbedding, imagePaths, facialArea, embeddingCount) VALUES (?, ?, ?, ?)", 
                               (embedding.tobytes(), json.dumps([image_path] if image_path else []), json.dumps(facial_area), 1))
        else:
            # Insert as the first person
            cursor.execute("INSERT INTO FaceGroups (averageEmbedding, imagePaths, facialArea, embeddingCount) VALUES (?, ?, ?, ?)", 
                           (embedding.tobytes(), json.dumps([image_path] if image_path else []), json.dumps(facial_area), 1))
    
    conn.commit()

facesUnprocessedMediaItems = get_unmapped_media_paths(baseDir)
if not facesUnprocessedMediaItems:
    exit(0)

# Database connection
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

# Create the FaceGroups table if it doesn't exist
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

# Process each new media item
for media_item in facesUnprocessedMediaItems:
    id,media_path, media_type = media_item
    if media_type == "image":
        process_new_image(media_path)
    elif media_type == "movie":
        extracted_frames = extract_frames_from_movie(media_path)
        for frame in extracted_frames:
            process_new_image(frame, image_path=media_path)  # Pass the original movie path as the reference

conn.close()
