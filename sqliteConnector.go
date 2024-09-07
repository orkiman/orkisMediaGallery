package main

import (
	"bytes"
	"crypto/md5"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3" // Import SQLite driver
	"gocv.io/x/gocv"
)

type ImageData struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

type VideoData struct {
	Width     int     `json:"width"`
	Height    int     `json:"height"`
	Duration  float64 `json:"duration"`
	FrameRate float64 `json:"frameRate"`
}

type MediaItem struct {
	MediaID            int64     `json:"mediaID"`  //unique
	FileName           string    `json:"fileName"` // unique name
	MediaType          string    `json:"mediaType"`
	LocalFilePath      string    `json:"filePath"`
	AbsoluteFilePath   string    `json:"absoluteFilePath"`
	LocalThumbnailPath string    `json:"thumbnailPath"`
	CreationDate       time.Time `json:"creationDate"`
	FileSize           int64     `json:"fileSize"`
	Checksum           string    `json:"checksum"`
	// ImageData          *ImageData `json:"imageData,omitempty"`
	// VideoData          *VideoData `json:"videoData,omitempty"`
}

func insertNewMediaToSqlDbAndGetNewMediaItem(db *sql.DB, absoluteMediaPath string) (MediaItem, error) {
	// todo : generate unique mediaID
	uniqueFileName, err := generateUniqueFileName(db, filepath.Base(absoluteMediaPath))
	if err != nil {
		return MediaItem{}, err
	}
	var mediaType string
	if isImage(absoluteMediaPath) {
		mediaType = "image"
	} else if isVideoFile(absoluteMediaPath) {
		mediaType = "video"
	} else {
		mediaType = "unknown"
	}
	if mediaType == "unknown" {
		return MediaItem{}, errors.New("unknown file type")
	}
	var creationDate time.Time
	if mediaType == "video" {
		// get creation date from video metadata
		creationDate, _ = getvideoCreationTime(absoluteMediaPath)
	} else {
		tagNames := []string{"DateTimeOriginal"}
		exifNameValueMap, _ := getExifNameValueMap(absoluteMediaPath, tagNames)
		creationDateStr := exifNameValueMap["DateTimeOriginal"]
		creationDate, _ = time.Parse("2006:01:02 15:04:05", creationDateStr)
	}
	// if err != nil {
	// 	return err
	// }

	fileInfo, err := os.Stat(absoluteMediaPath)
	if err != nil {
		return MediaItem{}, err
	}
	fileSize := fileInfo.Size()

	checksum, err := calculateChecksum(absoluteMediaPath)
	if err != nil {
		return MediaItem{}, err
	}

	mediaItem := MediaItem{
		FileName:           uniqueFileName,
		MediaType:          mediaType,
		LocalFilePath:      filepath.Join(filepath.Base(mediaDir), uniqueFileName),
		LocalThumbnailPath: filepath.Join(filepath.Base(thumbnailDir), strings.TrimSuffix(uniqueFileName, filepath.Ext(uniqueFileName))+"_thumb.jpg"),
		AbsoluteFilePath:   absoluteMediaPath,
		CreationDate:       creationDate,
		FileSize:           fileSize,
		Checksum:           checksum,
		// ImageData:          nil,
		// VideoData:          nil,
	}
	// handleMediaItem(mediaItem)

	// insert media item to mediaItems table
	// insertStatement := "INSERT INTO mediaItems VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
	insertStatement := "INSERT INTO mediaItems (fileName, mediaType, localFilePath, absoluteFilePath, localThumbnailPath, creationDate, fileSize, checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING mediaID"
	// todo : json values?
	err = db.QueryRow(insertStatement,
		mediaItem.FileName,
		mediaItem.MediaType,
		mediaItem.LocalFilePath,
		mediaItem.AbsoluteFilePath,
		mediaItem.LocalThumbnailPath,
		mediaItem.CreationDate,
		mediaItem.FileSize,
		mediaItem.Checksum,
	).Scan(&mediaItem.MediaID)

	if err != nil {
		log.Panic(err)
	}
	return mediaItem, err

}

func isFileAllreadyExistsInSqlDb(db *sql.DB, newFilePath string) (alreadyExists bool, existName string) {
	newChecksum, err := calculateChecksum(newFilePath)
	if err != nil {
		return false, ""
	}
	result := db.QueryRow("SELECT fileName FROM mediaItems WHERE checksum=?", newChecksum)

	err = result.Scan(&existName)
	if err == nil {
		return true, existName
	}
	if err == sql.ErrNoRows {
		return false, ""
	}
	log.Panic(err)
	return true, ""

}

func generateUniqueFileName(db *sql.DB, originalFileName string) (string, error) {
	// if file extention is .heic - change it to .jpg
	if strings.HasSuffix(originalFileName, ".heic") {
		originalFileName = strings.TrimSuffix(originalFileName, ".heic") + ".jpg"
	}

	var uniqueFileName = originalFileName
	var counter = 0

	for {
		exists, err := fileNameExists(db, uniqueFileName)
		if err != nil {
			return "", err
		}
		if !exists {
			return uniqueFileName, nil
		}
		counter++
		ext := filepath.Ext(originalFileName)
		baseName := strings.TrimSuffix(originalFileName, ext)
		uniqueFileName = fmt.Sprintf("%s(%d)%s", baseName, counter, ext)
	}
}
func fileNameExists(db *sql.DB, fileName string) (bool, error) {
	result := db.QueryRow("SELECT * FROM mediaItems WHERE fileName = ?", fileName)
	var id int
	err := result.Scan(&id)
	if err == nil {
		return true, nil
	}
	if err == sql.ErrNoRows {
		return false, nil
	}
	return true, err
}

func insertMediaToFacesUnprocessedMediaItemsTable(db *sql.DB, mediaItem *MediaItem) error {
	// mediaPath := filepath.Join(rootDir, mediaItem.LocalFilePath)
	// mediaType := mediaItem.MediaType
	mediaID := mediaItem.MediaID
	stmt, err := db.Prepare("INSERT INTO facesUnprocessedMediaItems (mediaID) VALUES (?)")
	if err != nil {
		// log.Panic(err)
		return fmt.Errorf("prepare statement: %v", err)
	}
	defer stmt.Close()

	// Execute the SQL statement
	_, err = stmt.Exec(mediaID)
	if err != nil {
		log.Panic(err)
		return fmt.Errorf("execute statement: %v", err)
	}

	return nil
}

func createMediaItemsTable(db *sql.DB) error {
	// Define the SQL statement to create the table
	createTableSQL := `CREATE TABLE IF NOT EXISTS mediaItems (
		mediaID 			INTEGER PRIMARY KEY AUTOINCREMENT,	-- Primary key, unique identifier for each media item
		fileName 			TEXT NOT NULL,                    	-- Name of the media file
		mediaType 			TEXT NOT NULL,                   	-- Type of media (image or video)
		localFilePath 		TEXT NOT NULL,               		-- Path from rootDirto the media file on the local device
		absoluteFilePath	TEXT NOT NULL,               		-- absolute Path to the media file on the local device
		localThumbnailPath 	TEXT,                   			-- Path to the thumbnail image for the media
		creationDate 		DATETIME,                     		-- Date and time when the media was created
		fileSize 			INTEGER,                          	-- Size of the media file in bytes
		checksum 			TEXT                              	-- Checksum of the media file for integrity verification
	);`

	// Execute the SQL statement
	_, err := db.Exec(createTableSQL)
	if err != nil {
		return fmt.Errorf("failed to create table: %v", err)
	}

	return nil
}

func createFacesUnprocessedMediaItemsTable(db *sql.DB) error {
	// Define the SQL statement to create the table
	createTableSQL := `
    CREATE TABLE IF NOT EXISTS facesUnprocessedMediaItems (
        mediaID INTEGER PRIMARY KEY
    );`

	// Execute the SQL statement
	_, err := db.Exec(createTableSQL)
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

	// query := "SELECT mediaPath, facial_area, type FROM faceEmbeddings WHERE personID=?"
	// err := db.QueryRow(query, personID).Scan(&mediaPath, &facialAreaJSON)
	query := `
			SELECT absoluteFilePath, facial_area, mediaType 
			FROM faceEmbeddings 
			JOIN mediaItems 
			ON faceEmbeddings.mediaID = mediaItems.mediaID 
			WHERE personID = ?
			`
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

func testViewFaces(db *sql.DB) {
	defer db.Close()
	// personID := 3
	// faceImages, err := getFacesImagesByPersonID(facesDbSqlite, personID)
	faceImages, err := getAllFacesImages(db)

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

func deleteMedia(db *sql.DB, mediaID int) error {
	// delete files
	var absoluteFilePath, localThumbnailPath string

	err := db.QueryRow("SELECT absoluteFilePath ,localThumbnailPath FROM MediaItems WHERE mediaID = ?", mediaID).Scan(&absoluteFilePath, &localThumbnailPath)
	if err != nil {
		log.Fatal("Failed to query database:", err)
		return err
	}
	absoluteThumbnailPath := filepath.Join(rootDir, localThumbnailPath)
	// remove media file

	err = os.Remove(absoluteFilePath)
	if err != nil {
		log.Fatal("Failed to remove media file:", err)
		return err
	}
	// remove thumbnail file
	err = os.Remove(absoluteThumbnailPath)
	if err != nil {
		log.Fatal("Failed to remove thumbnail file:", err)
		return err
	}

	// remove from db

	// Delete from MediaItems
	_, err = db.Exec("DELETE FROM MediaItems WHERE mediaID = ?", mediaID)
	if err != nil {
		log.Printf("Error deleting from MediaItems: %v", err)
		return err
	}

	// Delete from faceEmbeddings
	_, err = db.Exec("DELETE FROM faceEmbeddings WHERE mediaID = ?", mediaID)
	if err != nil {
		log.Printf("Error deleting from faceEmbeddings: %v", err)
		return err
	}

	// Delete from facesUnprocessedMediaItems
	_, err = db.Exec("DELETE FROM facesUnprocessedMediaItems WHERE mediaID = ?", mediaID)
	if err != nil {
		log.Printf("Error deleting from facesUnprocessedMediaItems: %v", err)
		return err
	}

	return nil

}

func calculateChecksum(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	hash := md5.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}

	checksum := hex.EncodeToString(hash.Sum(nil))
	return checksum, nil
}
func printDatabaseLength(db *sql.DB) {
	result := db.QueryRow("SELECT COUNT(*) FROM MediaItems")

	var count int
	err := result.Scan(&count)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Database length: ", count)
}

func getMediaItemsFilteredAndSorted(db *sql.DB, page, pageSize int, sortBy, filterFileName, sortOrder, personID string) (mediaItems []MediaItem, totalFiles int, err error) {
	var orderBy string
	switch sortBy {
	case "name":
		orderBy = "ORDER BY fileName"
	case "date":
		orderBy = "ORDER BY creationDate"
	case "size":
		orderBy = "ORDER BY fileSize"
	default:
		orderBy = ""
	}

	if sortOrder == "desc" {
		orderBy += " DESC"
	} else {
		orderBy += " ASC"
	}

	where := "WHERE fileName LIKE ?"
	if personID != "" {
		where = ` JOIN faceEmbeddings ON mediaItems.mediaID = faceEmbeddings.mediaID WHERE fileName LIKE ? AND personID = ?	`
	}

	// Building the query
	query := `
        SELECT DISTINCT
			mediaItems.mediaID,
			fileName, 
			mediaType, 
			localFilePath, 
			absoluteFilePath, 
			localThumbnailPath, 
			creationDate, 
			fileSize, 
			checksum
        FROM mediaItems 
        ` + where + `
        ` + orderBy + `
        LIMIT ? OFFSET ?`
	// log.Print("query:", query)
	// Calculate the offset for pagination
	offset := (page - 1) * pageSize

	// Execute the query
	var rows *sql.Rows
	if personID != "" {
		rows, err = db.Query(query, "%"+filterFileName+"%", personID, pageSize, offset)
	} else {
		rows, err = db.Query(query, "%"+filterFileName+"%", pageSize, offset)
	}
	if err != nil {
		log.Println("Error executing query:", err)
		return nil, 0, err
	}
	if rows != nil {
		defer rows.Close()
	}

	// Scan the results into the mediaItems slice
	for rows.Next() {
		var mediaItem MediaItem
		err := rows.Scan(
			&mediaItem.MediaID,
			&mediaItem.FileName,
			&mediaItem.MediaType,
			&mediaItem.LocalFilePath,
			&mediaItem.AbsoluteFilePath,
			&mediaItem.LocalThumbnailPath,
			&mediaItem.CreationDate,
			&mediaItem.FileSize,
			&mediaItem.Checksum,
		)
		if err != nil {
			return nil, 0, err
		}
		mediaItems = append(mediaItems, mediaItem)
	}

	// Check for any errors encountered during iteration
	if err = rows.Err(); err != nil {
		return nil, 0, err
	}

	totalFiles = len(mediaItems)

	return mediaItems, totalFiles, nil
}

func getMediaItemFromSql(db *sql.DB, mediaID int) (MediaItem, error) {
	result := db.QueryRow("SELECT * FROM MediaItems WHERE mediaID=?", mediaID)

	var mediaItem MediaItem
	err := result.Scan(
		&mediaItem.MediaID,
		&mediaItem.FileName,
		&mediaItem.MediaType,
		&mediaItem.LocalFilePath,
		&mediaItem.AbsoluteFilePath,
		&mediaItem.LocalThumbnailPath,
		&mediaItem.CreationDate,
		&mediaItem.FileSize,
		&mediaItem.Checksum,
	)
	if err != nil {
		return MediaItem{}, err
	}
	return mediaItem, nil
}

func getFaceImagesBase64FromImage(img gocv.Mat, facialAreaJSON string) (string, error) {

	// Parse the facial area JSON
	var facialArea FacialArea
	if err := json.Unmarshal([]byte(facialAreaJSON), &facialArea); err != nil {
		return "", fmt.Errorf("failed to parse facial areas: %v", err)
	}

	// Ensure the facial area is within bounds
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

	// Crop the image based on the facial area
	croppedImg := img.Region(image.Rect(facialArea.X, facialArea.Y, facialArea.X+facialArea.W, facialArea.Y+facialArea.H))
	defer croppedImg.Close()

	// Convert the cropped face to an image.Image for encoding
	imgGo, err := croppedImg.ToImage()
	if err != nil {
		return "", fmt.Errorf("failed to convert Mat to image.Image: %v", err)
	}

	// Encode the image to JPEG and then to Base64
	var buf bytes.Buffer
	err = jpeg.Encode(&buf, imgGo, nil)
	if err != nil {
		return "", fmt.Errorf("failed to encode image: %v", err)
	}
	imgBase64Str := base64.StdEncoding.EncodeToString(buf.Bytes())

	// Return the face
	return imgBase64Str, nil
}

func getImageFromVideoPathOld(videoPath string, frameNUmber int) (gocv.Mat, error) {

	// Open the video file
	v, err := gocv.VideoCaptureFile(videoPath)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to open video file: %v", err)
	}
	defer v.Close()

	// Set the frame number
	v.Set(gocv.VideoCapturePosFrames, float64(frameNUmber))

	// Read the frame
	frame := gocv.NewMat()
	if ok := v.Read(&frame); !ok {
		return gocv.NewMat(), fmt.Errorf("failed to read frame")
	}
	defer frame.Close()

	return frame, nil
}

func getImageFromVideoPath(videoPath string, frameNumber int) (gocv.Mat, error) {
	// Open the video file
	v, err := gocv.VideoCaptureFile(videoPath)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to open video file: %v", err)
	}
	defer v.Close()

	// Set the frame number
	v.Set(gocv.VideoCapturePosFrames, float64(frameNumber))

	// Read the frame
	frame := gocv.NewMat()
	if ok := v.Read(&frame); !ok {
		return gocv.NewMat(), fmt.Errorf("failed to read frame")
	}

	return frame, nil
}

type Face struct {
	ImageData  string
	PersonID   string
	MediaCount string
}

func getOneImagePerPerson(db *sql.DB) ([]Face, error) {

	var faces []Face
	var personID int
	var absoluteFilePath string
	var facialAreaJSON string
	var mediaType string
	var videoFrameNumber int
	var mediaCount int

	query := `
		SELECT 
			p.personID,
			m.absoluteFilePath,
			fe.facial_area,
			m.mediaType,
			fe.videoFrameNumber,
			COUNT(fe.mediaID) AS mediaCount
		FROM 
			persons p
		JOIN 
			faceEmbeddings fe ON p.personID = fe.personID
		JOIN 
			mediaItems m ON fe.mediaID = m.mediaID
		GROUP BY 
			fe.personID
		ORDER BY 
			mediaCount DESC;
	`

	rows, err := db.Query(query)
	if err != nil {
		return nil, fmt.Errorf("failed to query database: %v", err)
	}
	defer rows.Close()

	for rows.Next() {
		if err := rows.Scan(&personID, &absoluteFilePath, &facialAreaJSON, &mediaType, &videoFrameNumber, &mediaCount); err != nil {
			return nil, fmt.Errorf("failed to scan row: %v", err)
		}

		var img gocv.Mat
		if mediaType == "image" {
			img = gocv.IMRead(absoluteFilePath, gocv.IMReadColor)
			if img.Empty() {
				return nil, fmt.Errorf("failed to read image from path %s", absoluteFilePath)
			}
		} else if mediaType == "video" {
			img, err = getImageFromVideoPath(absoluteFilePath, videoFrameNumber)
			if err != nil {
				return nil, fmt.Errorf("failed to get image from video at path %s, frame %d: %v", absoluteFilePath, videoFrameNumber, err)
			}
		}

		// Make sure to close the Mat after processing
		defer img.Close()
		// showImage(img)
		face, err := getFaceImagesBase64FromImage(img, facialAreaJSON)
		if err != nil {
			return nil, fmt.Errorf("failed to get face image data for personID %d: %v", personID, err)
		}

		faces = append(faces, Face{
			ImageData:  face,
			PersonID:   strconv.Itoa(personID),
			MediaCount: strconv.Itoa(mediaCount),
		})
	}

	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("error encountered while iterating rows: %v", err)
	}

	return faces, nil
}

func getOneImagePerPersonWithoutPersonsTableWithoutProgressPrint(db *sql.DB) ([]Face, error) {

	var faces []Face
	var personID int
	var absoluteFilePath string
	var facialAreaJSON string
	var mediaType string
	var videoFrameNumber int
	var mediaCount int

	query := `
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
			`

	rows, err := db.Query(query)
	if err != nil {
		return nil, fmt.Errorf("failed to query database: %v", err)
	}
	defer rows.Close()

	for rows.Next() {
		if err := rows.Scan(&personID, &absoluteFilePath, &facialAreaJSON, &mediaType, &videoFrameNumber, &mediaCount); err != nil {
			return nil, fmt.Errorf("failed to scan row: %v", err)
		}

		var img gocv.Mat
		if mediaType == "image" {
			img = gocv.IMRead(absoluteFilePath, gocv.IMReadColor)
			if img.Empty() {
				return nil, fmt.Errorf("failed to read image from path %s", absoluteFilePath)
			}
		} else if mediaType == "video" {
			img, err = getImageFromVideoPath(absoluteFilePath, videoFrameNumber)
			if err != nil {
				return nil, fmt.Errorf("failed to get image from video at path %s, frame %d: %v", absoluteFilePath, videoFrameNumber, err)
			}
		}

		// Make sure to close the Mat after processing
		defer img.Close()
		// showImage(img)
		face, err := getFaceImagesBase64FromImage(img, facialAreaJSON)
		if err != nil {
			return nil, fmt.Errorf("failed to get face image data for personID %d: %v", personID, err)
		}

		faces = append(faces, Face{
			ImageData:  face,
			PersonID:   strconv.Itoa(personID),
			MediaCount: strconv.Itoa(mediaCount),
		})
	}

	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("error encountered while iterating rows: %v", err)
	}

	return faces, nil
}

func getOneImagePerPersonWithoutPersonsTable(db *sql.DB) ([]Face, error) {
	var faces []Face
	var personID int
	var absoluteFilePath string
	var facialAreaJSON string
	var mediaType string
	var videoFrameNumber int
	var mediaCount int

	// Count total rows to display progress
	countQuery := `
		SELECT COUNT(DISTINCT personID) 
		FROM faceEmbeddings;
	`
	var totalRows int
	err := db.QueryRow(countQuery).Scan(&totalRows)

	if err != nil {
		return nil, fmt.Errorf("failed to count rows: %v", err)
	}

	// Main query to get the data
	query := `
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
	`

	rows, err := db.Query(query)
	if err != nil {
		return nil, fmt.Errorf("failed to query database: %v", err)
	}
	defer rows.Close()

	// Initialize a counter for progress tracking
	currentRow := 0

	for rows.Next() {
		if err := rows.Scan(&personID, &absoluteFilePath, &facialAreaJSON, &mediaType, &videoFrameNumber, &mediaCount); err != nil {
			return nil, fmt.Errorf("failed to scan row: %v", err)
		}

		var img gocv.Mat
		if mediaType == "image" {
			img = gocv.IMRead(absoluteFilePath, gocv.IMReadColor)
			if img.Empty() {
				return nil, fmt.Errorf("failed to read image from path %s", absoluteFilePath)
			}
		} else if mediaType == "video" {
			img, err = getImageFromVideoPath(absoluteFilePath, videoFrameNumber)
			if err != nil {
				return nil, fmt.Errorf("failed to get image from video at path %s, frame %d: %v", absoluteFilePath, videoFrameNumber, err)
			}
		}

		defer img.Close()
		face, err := getFaceImagesBase64FromImage(img, facialAreaJSON)
		if err != nil {
			return nil, fmt.Errorf("failed to get face image data for personID %d: %v", personID, err)
		}

		faces = append(faces, Face{
			ImageData:  face,
			PersonID:   strconv.Itoa(personID),
			MediaCount: strconv.Itoa(mediaCount),
		})

		// Update progress
		currentRow++
		progress := float64(currentRow) / float64(totalRows) * 100
		fmt.Printf("Progress: %.2f%% (%d/%d)\n", progress, currentRow, totalRows)
	}

	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("error encountered while iterating rows: %v", err)
	}

	return faces, nil
}

func showImage(img gocv.Mat) {
	originalWidth := img.Cols()
	originalHeight := img.Rows()

	// Calculate the new dimensions, keeping the aspect ratio
	newWidth := 200
	newHeight := (newWidth * originalHeight) / originalWidth

	// Resize the image
	resizedImg := gocv.NewMat()
	defer resizedImg.Close()
	gocv.Resize(img, &resizedImg, image.Pt(newWidth, newHeight), 0, 0, gocv.InterpolationLinear)

	// Create a window to display the resized image
	window := gocv.NewWindow("Resized Image")
	defer window.Close()

	// Show the resized image in the window
	window.IMShow(resizedImg)

	// Wait until a key is pressed
	gocv.WaitKey(0)
}
