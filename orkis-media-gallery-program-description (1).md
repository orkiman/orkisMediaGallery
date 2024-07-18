# Orkis Media Gallery Application Description

## 1. Introduction

This document outlines the structure and functionality of a media gallery application designed to manage and display various types of media files, including images and videos. The application provides a user-friendly interface for uploading, viewing, and organizing media content.

## 2. System Architecture

### 2.1 Directory Structure

```
root/
|-- upload/
|-- media/
|-- thumbnails/
|-- heic/
|-- database.json
```

- `upload/`: Directory for manual file uploads
- `media/`: Processed and organized media files
- `thumbnails/`: Generated thumbnails for quick loading
- `heic/`: Original HEIC format images
- `database.json`: Metadata storage for all media items

### 2.2 Front-end

The application's front-end is browser-based, utilizing HTML and potentially other web technologies for a responsive and interactive user interface.

### 2.3 Back-end

The back-end is written in Go (Golang). It processes uploads, generates thumbnails, manages the database, and serves content to the front-end.

## 3. Key Features

### 3.1 File Upload

- Manual upload: Users can manually add files to the `upload/` directory.
- GUI upload: The application provides an interface for uploading files directly through the browser.

### 3.2 File Processing

- Thumbnail generation: Creates thumbnails for quick loading in the gallery view.
- HEIC conversion: Converts HEIC images to JPG for browser compatibility, storing originals in the `heic/` folder.
- Duplicate handling: Adds a number to filenames to ensure uniqueness.
- Metadata extraction: Extracts and stores EXIF and other relevant metadata.

### 3.3 Gallery View

- Paginated display: Shows a limited number of items per page.
- Mixed media: Displays both images and videos in a single view.
- Thumbnail differentiation: Video thumbnails have a distinct frame color for easy identification.

### 3.4 Sorting and Filtering

- Implements various sorting options (e.g., by date, name, size).
- Allows filtering based on media type, tags, or other metadata.

### 3.5 Media Interaction

#### Images:
- First click: Displays a fitted-size image.
- Subsequent clicks: Toggles between full-size view (centered on click position) and fitted view.
- Info button: Reveals all EXIF information for the image.

#### Videos:
- Thumbnail click: Plays the video fitted to the screen.

## 4. User Interface

### 4.1 Home Page

- Displays the media gallery with thumbnails.
- Includes sorting and filtering controls.
- Shows pagination controls.

### 4.2 Upload Interface

- Provides a drag-and-drop area or file selection button for uploading.
- Displays upload progress and confirmation.

### 4.3 Media Viewer

- Image viewer with zoom and pan capabilities.
- Video player with standard controls.
- Information panel for displaying metadata.

## 5. Data Management

### 5.1 Database Structure

The database file (using bboldDb) stores metadata for each media item, including:

- File name (Unique identifier)
- Media type (image/video)
- File path
- Thumbnail path
- Creation date
- File size
- checksum?
- user Tags
- Custom albums or categories

### 5.2 File Organization

- Uploaded files are processed and moved from `upload/` to `media/`.
- Thumbnails are generated and stored in `thumbnails/`.
- Original HEIC files are stored in `heic/`.

## 6. Future Enhancements

- Implement user accounts and access controls.
- Add sharing capabilities for media items or albums.
- Integrate with cloud storage services.
- Implement facial recognition for photo organization.
- Duplicate media detection based on content analysis rather than filename.

## 7. Conclusion

This media gallery application provides a comprehensive solution for managing and viewing various media types. Its modular design allows for easy expansion and customization to meet evolving user needs. The combination of a Go backend with a browser-based frontend ensures efficient processing and a user-friendly experience.
