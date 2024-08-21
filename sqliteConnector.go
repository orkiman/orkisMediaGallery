package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"image"
	"log"
	"math"
	"path/filepath"

	_ "github.com/mattn/go-sqlite3" // Import SQLite driver
	"gocv.io/x/gocv"
)

func insertMediaToFacesUnprocessedMediaItemsTable(mediaItem *MediaItem, facesDbSqlite *sql.DB) error {
	mediaPath := filepath.Join(rootDir, mediaItem.LocalFilePath)
	mediaType := mediaItem.MediaType

	stmt, err := facesDbSqlite.Prepare("INSERT INTO facesUnprocessedMediaItems (mediaItemPath, type) VALUES (?, ?)")
	if err != nil {
		return fmt.Errorf("prepare statement: %v", err)
	}
	defer stmt.Close()

	// Execute the SQL statement
	_, err = stmt.Exec(mediaPath, mediaType)
	if err != nil {
		return fmt.Errorf("execute statement: %v", err)
	}

	return nil
}

func createFacesUnprocessedMediaItemsTable(facesDbSqlite *sql.DB) error {
	// Define the SQL statement to create the table
	createTableSQL := `
    CREATE TABLE IF NOT EXISTS facesUnprocessedMediaItems (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mediaItemPath TEXT,
        type TEXT
    );`

	// Execute the SQL statement
	_, err := facesDbSqlite.Exec(createTableSQL)
	if err != nil {
		return fmt.Errorf("failed to create table: %v", err)
	}

	return nil
}

// Define a structure for facial area coordinates
type FacialArea struct {
	X int `json:"x"`
	Y int `json:"y"`
	W int `json:"w"`
	H int `json:"h"`
}

func getAllFacesImages(db *sql.DB) ([]gocv.Mat, error) {
	query := "SELECT COUNT(*) FROM persons"
	result := db.QueryRow(query)
	var count int
	if err := result.Scan(&count); err != nil {
		return nil, fmt.Errorf("failed to scan result: %v", err)
	}

	var faceImages []gocv.Mat
	for i := 1; i <= count; i++ {
		// Get face images for the current person ID
		personFaces, err := getFacesImagesByPersonID(db, i)
		if err != nil {
			return nil, fmt.Errorf("failed to get faces for person ID %d: %v", i, err)
		}

		// Append all face images for this person
		faceImages = append(faceImages, personFaces...)

		// Append an empty green image as a separator
		separator := gocv.NewMatWithSize(100, 100, gocv.MatTypeCV8UC3)
		separator.SetTo(gocv.NewScalar(0, 255, 0, 0)) // Green color
		faceImages = append(faceImages, separator)
	}

	return faceImages, nil
}

func getFacesImagesByPersonID(db *sql.DB, personID int) ([]gocv.Mat, error) {
	// Query the database for image paths and facial areas

	query := "SELECT mediaPath, facial_area, type FROM face_embeddings WHERE personID=?"
	// err := db.QueryRow(query, personID).Scan(&mediaPath, &facialAreaJSON)
	rows, err := db.Query(query, personID)

	if err != nil {
		log.Fatal(err)
		return nil, fmt.Errorf("failed to query database: %v", err)
	}
	defer rows.Close()

	var faceImages []gocv.Mat
	for rows.Next() {
		var mediaPath, facialAreaJSON, mediaType string
		if err := rows.Scan(&mediaPath, &facialAreaJSON, &mediaType); err != nil {
			log.Fatal(err)
			return nil, fmt.Errorf("failed to scan row: %v", err)
		}
		// todo : handle videos
		if mediaType != "image" {
			continue
		}

		// Parse JSON arrays
		var facialArea FacialArea
		if err := json.Unmarshal([]byte(facialAreaJSON), &facialArea); err != nil {
			log.Fatal(err)
			return nil, fmt.Errorf("failed to parse facial areas: %v", err)
		}

		img := gocv.IMRead(mediaPath, gocv.IMReadColor)
		if img.Empty() {
			return nil, fmt.Errorf("failed to load image from %s", mediaPath)
		}
		defer img.Close()

		// Crop the image based on the facial area
		// Ensure the facial area is within the image bounds
		if facialArea.X < 0 {
			facialArea.X = 0
		}
		if facialArea.Y < 0 {
			facialArea.Y = 0
		}
		if facialArea.X+facialArea.W > img.Cols() {
			facialArea.W = img.Cols() - facialArea.X
		}
		if facialArea.Y+facialArea.H > img.Rows() {
			facialArea.H = img.Rows() - facialArea.Y
		}

		croppedImg := img.Region(image.Rect(facialArea.X, facialArea.Y, facialArea.X+facialArea.W, facialArea.Y+facialArea.H))
		defer croppedImg.Close()

		// Resize the cropped face to 100 pixels height
		resizedFace := gocv.NewMat()
		defer resizedFace.Close() // Use defer to ensure it gets released later
		// Calculate the proportional height based on the desired height
		// newHeight := (facialArea.H * 100) / facialArea.W
		height := 150
		newWidth := (facialArea.W * height) / facialArea.H
		// gocv.Resize(croppedImg, &resizedFace, image.Point{X: 100, Y: newHeight}, 0, 0, gocv.InterpolationLinear)
		gocv.Resize(croppedImg, &resizedFace, image.Point{X: newWidth, Y: height}, 0, 0, gocv.InterpolationLinear)
		// Append the resized face image to the list
		faceImages = append(faceImages, resizedFace.Clone()) // Clone to keep the resized Mat alive
	}

	return faceImages, nil
}

// Function to create and display images in a grid

func displayImagesInGrid(faceImages []gocv.Mat) error {
	if len(faceImages) == 0 {
		fmt.Println("No images to display")
		return nil
	}

	// Calculate the grid dimensions
	numImages := len(faceImages)
	cols := int(math.Ceil(math.Sqrt(float64(numImages)))) // Calculate number of columns
	rows := (numImages + cols - 1) / cols                 // Ceil division

	// Increase the cell size for better visibility
	cellSize := 150 // Increased from the original 100

	// Create a blank image for the grid
	gridImg := gocv.NewMatWithSize(rows*cellSize, cols*cellSize, gocv.MatTypeCV8UC3)
	gridImg.SetTo(gocv.NewScalar(255, 255, 255, 0)) // Fill with white background

	for i, img := range faceImages {
		col := i % cols
		row := i / cols

		// Calculate aspect ratio to maintain proportions
		originalSize := img.Size()
		aspectRatio := float64(originalSize[1]) / float64(originalSize[0])

		// Determine the size of the image within the cell while maintaining aspect ratio
		var resizedWidth, resizedHeight int
		if aspectRatio > 1 {
			// Image is wider than tall
			resizedWidth = cellSize
			resizedHeight = int(float64(cellSize) / aspectRatio)
		} else {
			// Image is taller than wide or square
			resizedHeight = cellSize
			resizedWidth = int(float64(cellSize) * aspectRatio)
		}

		// Resize the image
		resizedImg := gocv.NewMat()
		gocv.Resize(img, &resizedImg, image.Point{X: resizedWidth, Y: resizedHeight}, 0, 0, gocv.InterpolationLinear)
		defer resizedImg.Close()

		// Calculate the top-left corner to center the image within the cell
		xOffset := (cellSize - resizedWidth) / 2
		yOffset := (cellSize - resizedHeight) / 2

		// Calculate the region in the grid image where the resized image will be placed
		rect := image.Rect(col*cellSize+xOffset, row*cellSize+yOffset, col*cellSize+xOffset+resizedWidth, row*cellSize+yOffset+resizedHeight)
		roi := gridImg.Region(rect)
		defer roi.Close()

		// Copy the resized image into the grid image
		resizedImg.CopyTo(&roi)
	}

	// Create a resizable window
	window := gocv.NewWindow("Face Images Grid")
	defer window.Close()

	// Set the initial window size to be larger
	initialWidth := cols * cellSize
	initialHeight := rows * cellSize
	window.ResizeWindow(initialWidth, initialHeight)

	// Display the grid image
	for {
		window.IMShow(gridImg)
		key := gocv.WaitKey(1)
		if key == 27 { // ESC key
			break
		}
	}

	return nil
}

// Example usage

func testViewFaces() {
	facesDbSqlite, err := sql.Open("sqlite3", "./faces.db")
	defer facesDbSqlite.Close()
	// personID := 3
	// faceImages, err := getFacesImagesByPersonID(facesDbSqlite, personID)
	faceImages, err := getAllFacesImages(facesDbSqlite)

	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Display the face images in a grid
	displayImagesInGrid(faceImages)

	// // Display the face images using GoCV
	// window := gocv.NewWindow("Face Images")
	// defer window.Close()
	// for i, faceImage := range faceImages {
	// 	window.IMShow(faceImage)
	// 	gocv.WaitKey(0) // Wait for a key press to display the next image
	// 	fmt.Printf("Displayed face image %d\n", i+1)
	// 	faceImage.Close() // Close each Mat after usage
	// }
}
