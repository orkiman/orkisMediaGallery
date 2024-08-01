import hdbscan
import numpy as np
import sqlite3

# Face embeddings with image IDs
embeddings = np.array([
    [0.234, 0.123, 0.456, 0.789, 0.012, 0.345, 0.678, 0.901, 0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, 0.888],
    [0.239, 0.135, 0.471, 0.802, 0.015, 0.359, 0.691, 0.914, 0.116, 0.227, 0.339, 0.451, 0.563, 0.675, 0.787, 0.899],
    [0.250, 0.150, 0.480, 0.820, 0.020, 0.370, 0.700, 0.930, 0.120, 0.230, 0.340, 0.450, 0.560, 0.670, 0.780, 0.890],
    [0.260, 0.160, 0.490, 0.840, 0.025, 0.380, 0.710, 0.950, 0.130, 0.240, 0.350, 0.460, 0.570, 0.680, 0.790, 0.900]
])

image_ids = ['image_001.jpg', 'image_002.jpg', 'image_003.jpg', 'image_004.jpg']

# Create an HDBSCAN clusterer
clusterer = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=2)

# Fit the clusterer to the embeddings
clusterer.fit(embeddings)

# Get the cluster labels
labels = clusterer.labels_

# Create a dictionary to store the clusters
clusters = {}
for i, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append({'image_id': image_ids[i], 'embedding': embeddings[i]})

# Connect to the SQLite database
conn = sqlite3.connect('face_clusters.db')
cursor = conn.cursor()

# Create the clusters table if it doesn't exist
cursor.execute(''' CREATE TABLE IF NOT EXISTS clusters ( cluster_id INTEGER, image_id TEXT, embedding BLOB ) ''')

# Insert the clusters into the database
for cluster_id, cluster in clusters.items():
    for image in cluster:
        cursor.execute(''' INSERT INTO clusters (cluster_id, image_id, embedding) VALUES (?, ?, ?) ''', (cluster_id, image['image_id'], image['embedding'].tobytes()))

# Commit the changes
conn.commit()

# Retrieve the clustered data from the database
cursor.execute(''' SELECT cluster_id, GROUP_CONCAT(image_id) AS image_ids FROM clusters GROUP BY cluster_id ''')

# Fetch the results
results = cursor.fetchall()

# Print the output
print("Clustered Data:")
for row in results:
    cluster_id, image_ids = row
    print(f'Cluster {cluster_id}: {image_ids}')

# Close the connection
conn.close()