## **Project Description: Self-Hosted Media Gallery**

**Purpose:**

This project is a self-hosted media gallery application that allows users to manage, view, and organize their personal media files (images and videos). It combines Go and Python components for efficient media processing, database interaction, and user interface.

**Key Features:**

* **Media Management:**  
  * Uploads and stores media files (images and videos).  
  * Generates thumbnails for images and videos.  
  * Organizes media based on various criteria (date, name, size).  
  * Provides search functionality.  
* **Facial Recognition:**  
  * Identifies faces in images and videos.  
  * Clusters faces based on similarity.  
  * Associates faces with individuals.  
* **User Interface:**  
  * Web-based interface for browsing and managing media.  
  * User authentication and authorization.  
* **Database Integration:**  
  * Uses a SQLite database to store media information, face embeddings, and user data.  
  * Efficiently manages database transactions.

**Technologies:**

* **Go:**  
  * Media file processing (uploading, storing, generating thumbnails).  
  * Database interactions (SQLite).  
  * Web server and API implementation.  
* **Python:**  
  * Facial recognition (using DeepFace).  
  * Clustering (using Chinese Whispers).  
* **SQLite:**  
  * Database storage for media information, face embeddings, and user data.  
* **HTML, CSS, JavaScript:**  
  * Frontend web development for the user interface.

**Workflow:**

1. **Media Upload:**  
   * Users upload media files (images and videos).  
   * The Go server processes the uploaded files, stores them in the appropriate directories, and generates thumbnails.  
2. **Facial Recognition:**  
   * The Python component extracts facial embeddings from images and videos.  
   * Clusters face embeddings based on similarity.  
   * Associates faces with individuals (if applicable).  
   * Stores facial data in the database.  
3. **Database Integration:**  
   * The Go application interacts with the SQLite database to store media information, face embeddings, and user data.  
   * Performs queries to retrieve and update data.  
4. **User Interface:**  
   * The web interface (built using HTML, CSS, and JavaScript) allows users to browse media, view thumbnails, search for files, and manage their gallery.

**Database Structure:**

* **mediaItems**: Stores information about media files (ID, path, type, creation date, checksum).  
* **facesUnprocessedMediaItems**: Tracks unprocessed media files for facial recognition.  
* **faceEmbeddings**: Stores facial embeddings, media IDs, facial areas, and video frame numbers.  
* **Persons**: Stores person IDs, average embeddings, and associated face embeddings.  
* **Users**: Stores user information (username, password, etc.).

**Additional Features:**

* **User Authentication:**  
  * Allows users to log in and access their personal media.  
* **Search Functionality:**  
  * Enables users to search for media files based on keywords or metadata.  
* **Sharing and Collaboration:**  
  * Provides options for sharing media with other users.

sql tables structure here : https://docs.google.com/document/d/1PrIiOg6oL\_UbHzmdwUrBh9yQDj1p-Xr8mwjz3YOX8Bk/edit?usp=sharing  
