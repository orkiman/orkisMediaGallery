import sqlite3
import json
import hdbscan
import numpy as np
from collections import defaultdict

# Fetch embeddings from SQLite

def fetch_embeddings():
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()
    cursor.execute("SELECT media_id, face_embedding FROM embeddings")
    
    embeddings = []
    media_ids = []
    
    for row in cursor.fetchall():
        media_id = row[0]
        face_embedding = json.loads(row[1])
        embeddings.append(face_embedding)
        media_ids.append(media_id)
        
    conn.close()
    return np.array(embeddings), media_ids

# Fetch embeddings
embeddings, media_ids = fetch_embeddings()

# Perform clustering with HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
cluster_labels = clusterer.fit_predict(embeddings)

# Prepare clustering output
cluster_dict = defaultdict(list)
for label, media_id in zip(cluster_labels, media_ids):
    if label != -1:  # Ignore noise points
        cluster_dict[label].append(media_id)

# Save clustering results to SQLite
conn = sqlite3.connect('embeddings.db')
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS clusters (cluster_id INTEGER PRIMARY KEY, media_ids TEXT)")
cursor.execute("DELETE FROM clusters")  # Clear previous clustering results

for cluster_id, media_ids in cluster_dict.items():
    media_ids_str = json.dumps(media_ids)
    cursor.execute("INSERT INTO clusters (cluster_id, media_ids) VALUES (?, ?)",
                   (cluster_id, media_ids_str))

conn.commit()
conn.close()

print("Clustering results saved successfully")