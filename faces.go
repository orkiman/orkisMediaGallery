package main

import (
	"fmt"
	"image"
	"image/color"

	"gocv.io/x/gocv"
)

func ProcessMediaItem(item *MediaItem) error {
	if item.MediaType == "image" {
		return processImage(item)
	} else if item.MediaType == "video" {
		return processVideo(item)
	}
	return fmt.Errorf("unsupported media type: %s", item.MediaType)
}

func processImage(item *MediaItem) error {
	img := gocv.IMRead(item.LocalFilePath, gocv.IMReadColor)
	if img.Empty() {
		return fmt.Errorf("error reading image: %s", item.LocalFilePath)
	}
	defer img.Close()

	faces := DetectFaces(img)

	// Update ImageData
	item.ImageData = &ImageData{
		Width:  img.Cols(),
		Height: img.Rows(),
	}

	// Generate face embeddings and store them
	for _, face := range faces {
		faceID, err := generateAndStoreFaceEmbedding(img, face)
		if err != nil {
			return fmt.Errorf("error processing face: %v", err)
		}
		item.FaceIDs = append(item.FaceIDs, faceID)
	}

	fmt.Printf("Processed image %s, found %d faces\n", item.FileName, len(faces))
	return nil
}

func processVideo(item *MediaItem) error {
	video, err := gocv.VideoCaptureFile(item.LocalFilePath)
	if err != nil {
		return fmt.Errorf("error opening video: %v", err)
	}
	defer video.Close()

	fps := video.Get(gocv.VideoCaptureFPS)
	frameInterval := int(fps) * 5 // Process a frame every 5 seconds
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

	processedFaces := make(map[string]bool)

	for frameCount := 0; frameCount < totalFrames; frameCount++ {
		if ok := video.Read(&frame); !ok {
			if frame.Empty() {
				break
			}
			return fmt.Errorf("error reading frame %d", frameCount)
		}

		if frameCount%frameInterval == 0 {
			faces := DetectFaces(frame)

			for _, face := range faces {
				faceID, err := generateAndStoreFaceEmbedding(frame, face)
				if err != nil {
					return fmt.Errorf("error processing face in frame %d: %v", frameCount, err)
				}
				if !processedFaces[faceID] {
					item.FaceIDs = append(item.FaceIDs, faceID)
					processedFaces[faceID] = true
				}
			}

			fmt.Printf("Processed frame %d of video %s, found %d faces\n", frameCount, item.FileName, len(faces))
		}
	}

	return nil
}

func DetectFaces(img gocv.Mat) []image.Rectangle {
	// Load the pre-trained Haar Cascade classifier for face detection
	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()

	if !classifier.Load("haarcascade_frontalface_default.xml") {
		panic("Error loading face detection classifier")
	}

	// Detect faces
	faces := classifier.DetectMultiScale(img)

	return faces
}

func generateAndStoreFaceEmbedding(img gocv.Mat, face image.Rectangle) (string, error) {
	// TODO: Implement face embedding generation
	// TODO: Store face embedding in Milvus
	// TODO: Generate and return a unique faceID
	return "dummy-face-id", nil
}

func testCv() {
	// Read the input image
	img := gocv.IMRead("1.jpg", gocv.IMReadColor)
	if img.Empty() {
		fmt.Println("Error reading image")
		return
	}
	defer img.Close()

	// Load the DNN model
	net := gocv.ReadNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
	if net.Empty() {
		fmt.Println("Error reading network model")
		return
	}
	defer net.Close()

	// Prepare the image for the network
	blob := gocv.BlobFromImage(img, 1.0, image.Pt(300, 300), gocv.NewScalar(104, 177, 123, 0), false, false)
	defer blob.Close()

	// Set the input to the network
	net.SetInput(blob, "")

	// Run a forward pass through the network
	detections := net.Forward("")
	defer detections.Close()

	// Process the detections
	for i := 0; i < detections.Total(); i += 7 {
		confidence := detections.GetFloatAt(0, i+2)
		if confidence > 0.5 {
			left := int(detections.GetFloatAt(0, i+3) * float32(img.Cols()))
			top := int(detections.GetFloatAt(0, i+4) * float32(img.Rows()))
			right := int(detections.GetFloatAt(0, i+5) * float32(img.Cols()))
			bottom := int(detections.GetFloatAt(0, i+6) * float32(img.Rows()))

			// Draw rectangle around face
			gocv.Rectangle(&img, image.Rect(left, top, right, bottom), color.RGBA{0, 255, 0, 0}, 2)
		}
	}

	// Save the output image
	gocv.IMWrite("output.jpg", img)

	fmt.Println("Face detection completed. Check output.jpg for results.")
}
