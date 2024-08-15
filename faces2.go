package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"path/filepath"
	"strconv"
	"strings"

	_ "github.com/mattn/go-sqlite3"

	"github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

var embeddingsDb *sql.DB

func prepareInputForFaceRecognitionOriginal(face gocv.Mat) (*onnxruntime_go.Tensor[float32], error) {
	// Define the expected input size
	inputSize := image.Pt(112, 112)

	// Resize the image to 112x112 if necessary
	resizedFace := gocv.NewMat()
	defer resizedFace.Close()
	if face.Cols() != inputSize.X || face.Rows() != inputSize.Y {
		gocv.Resize(face, &resizedFace, inputSize, 0, 0, gocv.InterpolationLinear)
	} else {
		face.CopyTo(&resizedFace)
	}

	// Convert to float32 and normalize to range [0, 1]
	resizedFace.ConvertTo(&resizedFace, gocv.MatTypeCV32F)
	gocv.Normalize(resizedFace, &resizedFace, 0, 1, gocv.NormMinMax)

	// Split the image into separate channels
	channels := gocv.Split(resizedFace)
	defer func() {
		for _, channel := range channels {
			channel.Close()
		}
	}()

	// Create input tensor with shape (1, 3, 112, 112)
	inputShape := onnxruntime_go.NewShape(1, 3, 112, 112)
	inputTensor, err := onnxruntime_go.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	// Populate the tensor with image data
	inputData := inputTensor.GetData()
	channelSize := inputSize.X * inputSize.Y
	for i := 0; i < 3; i++ {
		channelData, err := channels[i].DataPtrFloat32()
		if err != nil {
			return nil, fmt.Errorf("failed to get channel data: %w", err)
		}
		copy(inputData[i*channelSize:(i+1)*channelSize], channelData)
	}

	return inputTensor, nil
}
func prepareInputForFaceRecognition(face gocv.Mat) (*onnxruntime_go.Tensor[float32], error) {
	// Define the expected input size
	inputSize := image.Pt(112, 112)

	// Resize the image to 112x112 if necessary
	resizedFace := gocv.NewMat()
	defer resizedFace.Close()
	if face.Cols() != inputSize.X || face.Rows() != inputSize.Y {
		gocv.Resize(face, &resizedFace, inputSize, 0, 0, gocv.InterpolationLinear)
	} else {
		face.CopyTo(&resizedFace)
	}

	// Convert to float32
	resizedFace.ConvertTo(&resizedFace, gocv.MatTypeCV32F)

	// Normalize to range [-1, 1]
	normalizedFace := gocv.NewMat()
	defer normalizedFace.Close()
	resizedFace.ConvertTo(&normalizedFace, gocv.MatTypeCV32F)
	normalizedFace.DivideFloat(127.5)
	normalizedFace.SubtractFloat(1.0)

	// Split the image into separate channels (BGR order)
	channels := gocv.Split(normalizedFace)
	defer func() {
		for _, channel := range channels {
			channel.Close()
		}
	}()

	// Create input tensor with shape (1, 3, 112, 112)
	inputShape := onnxruntime_go.NewShape(1, 3, 112, 112)
	inputTensor, err := onnxruntime_go.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	// Populate the tensor with image data
	inputData := inputTensor.GetData()
	channelSize := inputSize.X * inputSize.Y
	for i := 0; i < 3; i++ {
		channelData, err := channels[i].DataPtrFloat32()
		if err != nil {
			return nil, fmt.Errorf("failed to get channel data: %w", err)
		}
		copy(inputData[i*channelSize:(i+1)*channelSize], channelData)
	}

	return inputTensor, nil
}

func generateEmbedding(face gocv.Mat) ([]float32, error) {
	onnxruntime_go.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.so")
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	// Prepare input tensor
	inputTensor, err := prepareInputForFaceRecognition(face)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	// Define input and output names
	inputNames := []string{"data"}
	outputNames := []string{"fc1"}

	// Create output tensor placeholder
	outputShape := onnxruntime_go.NewShape(1, 512)
	outputTensor, err := onnxruntime_go.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Create a new ONNX Runtime session
	session, err := onnxruntime_go.NewSession[float32](
		"staticFiles/aiModels/imageRecognition/arcfaceresnet100-8.onnx",
		inputNames,
		outputNames,
		[]*onnxruntime_go.Tensor[float32]{inputTensor},
		[]*onnxruntime_go.Tensor[float32]{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}
	defer session.Destroy()

	// Run the model
	err = session.Run()
	if err != nil {
		return nil, fmt.Errorf("failed to run model: %w", err)
	}

	// Extract the embedding from the output tensor
	embedding := outputTensor.GetData()

	return embedding, nil
}

func generateAndStoreFaceEmbedding(img gocv.Mat, face image.Rectangle, mediaID string) error {
	faceROI := img.Region(face)
	defer faceROI.Close()

	// Save the image immediately instead of deferring
	outPath := fmt.Sprintf("face_%s.jpg", mediaID)
	success := gocv.IMWrite(outPath, faceROI)
	if !success {
		return fmt.Errorf("failed to save image: %s", outPath)
	}

	// Preprocess: resize to 112x112, convert to float32, normalize
	processedFace := gocv.NewMat()
	defer processedFace.Close()
	gocv.Resize(faceROI, &processedFace, image.Pt(112, 112), 0, 0, gocv.InterpolationDefault)
	processedFace.ConvertTo(&processedFace, gocv.MatTypeCV32F)
	gocv.Normalize(processedFace, &processedFace, 0, 1, gocv.NormMinMax)

	// Generate embedding
	embedding, err := generateEmbedding(processedFace)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}
	embedding = normalizeEmbedding(embedding)
	return insertEmbedding(mediaID, embedding)

}

func normalizeEmbedding(embedding []float32) []float32 {
	var sum float32
	for _, v := range embedding {
		sum += v * v
	}
	magnitude := float32(math.Sqrt(float64(sum)))
	normalized := make([]float32, len(embedding))
	for i, v := range embedding {
		normalized[i] = v / magnitude
	}
	return normalized
}

func ProcessMediaItemFaces(item *MediaItem) {
	if item.MediaType == "image" {
		img := gocv.IMRead(filepath.Join(mediaDir, item.FileName), gocv.IMReadColor)
		processImage(img, item.FileName)
	} else if item.MediaType == "video" {
		processVideo(item)
	}
	// return fmt.Errorf("unsupported media type: %s", item.MediaType)
}

func processVideo(item *MediaItem) error {
	video, err := gocv.VideoCaptureFile(item.LocalFilePath)
	if err != nil {
		return fmt.Errorf("error opening video: %v", err)
	}
	defer video.Close()

	fps := video.Get(gocv.VideoCaptureFPS)
	frameInterval := int(fps) * 2 // Process a frame every 2 seconds
	totalFrames := int(video.Get(gocv.VideoCaptureFrameCount))

	// Update VideoData
	item.VideoData = &VideoData{
		Width:     int(video.Get(gocv.VideoCaptureFrameWidth)),
		Height:    int(video.Get(gocv.VideoCaptureFrameHeight)),
		Duration:  float64(totalFrames) / fps,
		FrameRate: fps,
	}

	frame := gocv.NewMat()
	defer frame.Close()

	// processedFaces := make(map[string]bool)
	//extract image from video every interval and call processimage

	for frameCount := 0; frameCount < totalFrames; frameCount++ {
		if frameCount%frameInterval == 0 {
			if ok := video.Read(&frame); !ok {
				if frame.Empty() {
					break
				}
				return fmt.Errorf("error reading frame %d", frameCount)
			}
			processImage(frame, item.FileName)
		}
	}
	return nil
}

func processImage(img gocv.Mat, mediaID string) {
	// Your face detection code...
	// img := gocv.IMRead(filepath.Join(mediaDir, mediaItem.FileName), gocv.IMReadColor)
	if img.Empty() {
		fmt.Println("Error reading image")
		return
	}
	defer img.Close()
	faces, err := getFaceRects(img)
	if err != nil {
		fmt.Printf("Error detecting faces: %v\n", err)
		return
	}

	// Assume we have detected a face rectangle
	for _, faceRect := range faces {

		// Generate and store face embedding
		err := generateAndStoreFaceEmbedding(img, faceRect, mediaID)
		if err != nil {
			fmt.Printf("Error generating and storing face embedding: %v\n", err)
			return
		}

		// fmt.Printf("Generated face embedding with ID: %s\n", faceID)
	}
}

func getFaceRects(img gocv.Mat) ([]image.Rectangle, error) {
	net := gocv.ReadNetFromCaffe("staticFiles/aiModels/imageDetection/deploy.prototxt",
		"staticFiles/aiModels/imageDetection/res10_300x300_ssd_iter_140000.caffemodel")
	if net.Empty() {
		// fmt.Println("Error reading network model")
		return nil, fmt.Errorf("Error reading network model")
	}
	defer net.Close()

	blob := gocv.BlobFromImage(img, 1.0, image.Pt(300, 300), gocv.NewScalar(104, 177, 123, 0), false, false)
	defer blob.Close()

	net.SetInput(blob, "")
	detections := net.Forward("")
	defer detections.Close()
	var faceRects []image.Rectangle
	for i := 0; i < detections.Total(); i += 7 {
		confidence := detections.GetFloatAt(0, i+2)
		if confidence > 0.5 {
			left := int(detections.GetFloatAt(0, i+3) * float32(img.Cols()))
			top := int(detections.GetFloatAt(0, i+4) * float32(img.Rows()))
			right := int(detections.GetFloatAt(0, i+5) * float32(img.Cols()))
			bottom := int(detections.GetFloatAt(0, i+6) * float32(img.Rows()))

			faceRect := image.Rect(left, top, right, bottom)
			faceRects = append(faceRects, faceRect)

		}
	}

	// Draw rectangles around faces
	for _, faceRect := range faceRects {
		gocv.Rectangle(&img, faceRect, color.RGBA{0, 255, 0, 0}, 2)
	}
	gocv.IMWrite("output.jpg", img)

	return faceRects, nil
}

func openEmbeddingsDb() *sql.DB {
	// Open SQLite database
	// var err error
	embeddingsDb, err := sql.Open("sqlite3", "./embeddings.db")
	if err != nil {
		log.Fatal(err)
		return nil
	}
	// Create table if not exists
	statement, err := embeddingsDb.Prepare("CREATE TABLE IF NOT EXISTS embeddings (media_id TEXT, face_embedding BLOB)")
	if err != nil {
		log.Fatal(err)
	}
	statement.Exec()
	return embeddingsDb
}

// func closeEmbeddingsDb(clusteringDb *sql.DB) {
// 	// Close database
// 	err := clusteringDb.Close()
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// }

func insertEmbedding(mediaID string, embedding []float32) error {
	fmt.Println("inserting embedding: ", mediaID, " : ", embedding)
	// Convert embedding to byte array
	embeddingBytes, err := json.Marshal(embedding)
	if err != nil {
		return err
	}
	// Insert embedding into database
	_, err = embeddingsDb.Exec("INSERT INTO embeddings VALUES (?, ?)", mediaID, embeddingBytes)
	if err != nil {
		return err
	}
	fmt.Println("embedding inserted successfully")
	return nil
}

func getClusteringByFrequency() ([]struct {
	ClusterID int    "json:\"clusterID\""
	MediaIDs  string "json:\"mediaIDs\""
}, error) {
	embeddingsDb := openEmbeddingsDb()
	defer embeddingsDb.Close()
	// Prepare the SQL query
	query := `
		SELECT cluster_id, GROUP_CONCAT(media_ids) AS media_ids 
		FROM clusters 
		GROUP BY cluster_id
		ORDER BY COUNT(media_ids) DESC
	`
	// Execute the query
	rows, err := embeddingsDb.Query(query)
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	var result []struct {
		ClusterID int    `json:"clusterID"`
		MediaIDs  string `json:"mediaIDs"`
	}

	for rows.Next() {
		var clusterID int
		var mediaIDs string
		err := rows.Scan(&clusterID, &mediaIDs)
		if err != nil {
			log.Fatal(err)
		}
		result = append(result, struct {
			ClusterID int    `json:"clusterID"`
			MediaIDs  string `json:"mediaIDs"`
		}{ClusterID: clusterID, MediaIDs: mediaIDs})

	}
	if err := rows.Err(); err != nil {
		log.Fatal(err)
	}
	log.Println("getClusteringByFrequency: \n", result)
	return result, err

}

func getClustersAndUpdatefacesDataInMediaDb() {

	embeddingsDb := openEmbeddingsDb()
	defer embeddingsDb.Close()

	// Prepare the SQL query
	query := `
		SELECT cluster_id, GROUP_CONCAT(media_ids) AS media_ids 
		FROM clusters 
		GROUP BY cluster_id
		ORDER BY COUNT(media_ids) DESC
	`

	// Execute the query
	rows, err := embeddingsDb.Query(query)
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	// Print the output
	fmt.Println("Clustered Data:")
	for rows.Next() {
		var clusterID int
		var mediaIDs string
		err := rows.Scan(&clusterID, &mediaIDs)
		if err != nil {
			log.Fatal(err)
		}

		// Split the concatenated media IDs
		mediaIDList := strings.Split(mediaIDs, ",")
		for _, mediaId := range mediaIDList {
			var mediaItem *MediaItem
			mediaItem, err = getMediaItem(mediaDB, mediaId)

			if err != nil {
				log.Fatal(err)
			}
			mediaItem.FaceIDs = append(mediaItem.FaceIDs, strconv.Itoa(clusterID))

			updateMediaItem(mediaDB, mediaItem)
			// todo : update all photos with cluster idmediaId
			// maybe prepare a list of common clusters descending order
			// Update the faces data in the media table

		}

		fmt.Printf("Cluster %d: %s\n", clusterID, strings.Join(mediaIDList, ", "))
	}

	if err := rows.Err(); err != nil {
		log.Fatal(err)
	}
}
func testFaceDetection2() {
	getFaceRects(gocv.IMRead("/home/orkiman/vscodeProjects/python/pyFace2/facesDis/or4.jpg", gocv.IMReadColor))
}
