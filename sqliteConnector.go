package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"image"
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
	query := "SELECT COUNT(*) FROM FaceGroups"
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

// func getOneFaceForEachPerson(db *sql.DB) ([]gocv.Mat, error) {
// 	query := "SELECT PersonID, imagePathes FROM FaceGroups ORDER BY "

// getFacesImagesByPersonID retrieves face images for a given personID using GoCV
func getFacesImagesByPersonID(db *sql.DB, personID int) ([]gocv.Mat, error) {
	// Query the database for image paths and facial areas
	var imagePathsJSON, facialAreaJSON string
	query := "SELECT imagePaths, facialArea FROM FaceGroups WHERE personID=?"
	err := db.QueryRow(query, personID).Scan(&imagePathsJSON, &facialAreaJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to query database: %v", err)
	}

	// Parse JSON arrays
	var imagePaths []string
	var facialAreas []FacialArea
	if err := json.Unmarshal([]byte(imagePathsJSON), &imagePaths); err != nil {
		return nil, fmt.Errorf("failed to parse image paths: %v", err)
	}
	if err := json.Unmarshal([]byte(facialAreaJSON), &facialAreas); err != nil {
		return nil, fmt.Errorf("failed to parse facial areas: %v", err)
	}

	// Ensure image paths and facial areas match in length
	if len(imagePaths) != len(facialAreas) {
		return nil, fmt.Errorf("mismatched lengths: %d image paths, %d facial areas", len(imagePaths), len(facialAreas))
	}

	// Load and crop images based on facial areas
	var faceImages []gocv.Mat
	for i, path := range imagePaths {
		// Read the image using GoCV
		img := gocv.IMRead(path, gocv.IMReadColor)
		if img.Empty() {
			return nil, fmt.Errorf("failed to load image from %s", path)
		}
		defer img.Close()

		// Crop the image based on the facial area
		facialArea := facialAreas[i]

		// // Ensure the facial area is within the image bounds
		// if facialArea.X < 0 || facialArea.Y < 0 || facialArea.X+facialArea.W > img.Cols() || facialArea.Y+facialArea.H > img.Rows() {
		// 	fmt.Printf("Invalid facial area for image %s: %v\n", path, facialArea)
		// 	continue // Skip this image
		// }

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

		// Resize the cropped face to 100x100 pixels
		resizedFace := gocv.NewMat()
		defer resizedFace.Close() // Use defer to ensure it gets released later
		// Calculate the proportional height based on the desired width
		newHeight := (facialArea.H * 100) / facialArea.W
		gocv.Resize(croppedImg, &resizedFace, image.Point{X: 100, Y: newHeight}, 0, 0, gocv.InterpolationLinear)
		// Append the resized face image to the list
		faceImages = append(faceImages, resizedFace.Clone()) // Clone to keep the resized Mat alive
	}

	return faceImages, nil
}

const (
	gridWidth = 500 // Width of the entire grid image
	cellSize  = 100 // Size of each cell in the grid
)

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

		// Resize image to fit the cell size
		resizedImg := gocv.NewMat()
		gocv.Resize(img, &resizedImg, image.Point{X: cellSize, Y: cellSize}, 0, 0, gocv.InterpolationLinear)
		defer resizedImg.Close()

		// Calculate the region in the grid image where the resized image will be placed
		rect := image.Rect(col*cellSize, row*cellSize, (col+1)*cellSize, (row+1)*cellSize)
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
