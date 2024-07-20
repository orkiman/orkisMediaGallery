package main

import (
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"go.etcd.io/bbolt"
)

// - File name (Unique identifier)
// - Media type (image/video)
// - File path
// - Thumbnail path
// - Creation date
// - File size
// - checksum?
// - user Tags
// - Custom albums or categories

type MediaFile struct {
	FileName           string     `json:"fileName"`
	MediaType          string     `json:"mediaType"`
	LocalFilePath      string     `json:"filePath"`
	LocalThumbnailPath string     `json:"thumbnailPath"`
	CreationDate       time.Time  `json:"creationDate"`
	FileSize           int64      `json:"fileSize"`
	ImageData          *ImageData `json:"imageData,omitempty"`
	VideoData          *VideoData `json:"videoData,omitempty"`
	Checksum           string     `json:"checksum"`
	Tags               []string   `json:"tags"`
	Albums             []string   `json:"albums"`
}

type ImageData struct {
	Resolution string `json:"resolution"`
	EXIF       string `json:"exif"`
}

type VideoData struct {
	Duration   string `json:"duration"`
	Resolution string `json:"resolution"`
}

var db *bbolt.DB

func openDb() (*bbolt.DB, error) {
	var err error
	db, err = bbolt.Open("media.db", 0600, nil)
	if err != nil {
		log.Fatal(err)
		return nil, err
	}

	return db, nil
}

func insertMediaToDb(mediaPath string) error {
	uniqueFileName, err := generateUniqueFileName(db, filepath.Base(mediaPath))
	if err != nil {
		return err
	}
	var mediaType string
	if isImage(mediaPath) {
		mediaType = "image"
	} else if isVideoFile(mediaPath) {
		mediaType = "video"
	} else {
		mediaType = "unknown"
	}
	if mediaType == "unknown" {
		return errors.New("unknown file type")
	}
	var creationDate time.Time
	if mediaType == "video" {
		// get creation date from video metadata
		creationDate, _ = getvideoCreationTime(mediaPath)
	} else {
		tagNames := []string{"DateTimeOriginal"}
		exifNameValueMap, _ := getExifNameValueMap(mediaPath, tagNames)
		creationDateStr := exifNameValueMap["DateTimeOriginal"]
		creationDate, _ = time.Parse("2006:01:02 15:04:05", creationDateStr)
	}
	// if err != nil {
	// 	return err
	// }

	fileInfo, err := os.Stat(mediaPath)
	if err != nil {
		return err
	}
	fileSize := fileInfo.Size()

	checksum, err := calculateChecksum(mediaPath)
	if err != nil {
		return err
	}

	mediaFile := MediaFile{
		FileName:      uniqueFileName,
		MediaType:     mediaType,
		LocalFilePath: filepath.Join(filepath.Base(mediaDir), uniqueFileName),
		// FilePath:      filepath.Join(mediaDir, uniqueFileName),
		LocalThumbnailPath: filepath.Join(filepath.Base(thumbnailDir), strings.TrimSuffix(uniqueFileName, filepath.Ext(uniqueFileName))+"_thumb.jpg"),
		CreationDate:       creationDate,
		FileSize:           fileSize,
		ImageData:          nil,
		VideoData:          nil,
		Checksum:           checksum,
		Tags:               []string{},
		Albums:             []string{},
	}

	return db.Update(func(tx *bbolt.Tx) error {
		b, err := tx.CreateBucketIfNotExists([]byte("MediaFiles"))
		if err != nil {
			return err
		}

		encoded, err := json.Marshal(mediaFile)
		if err != nil {
			return err
		}

		return b.Put([]byte(mediaFile.FileName), encoded)
	})

}

func deleteMedia(fileName []string) error {
	return db.Update(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte("MediaFiles"))
		if bucket == nil {
			return nil
		}
		for _, name := range fileName {
			mediaFile, err := getMediaFile(db, name)
			if err != nil {
				return err
			}

			err = os.Rename(filepath.Join(mediaDir, name), filepath.Join(binDir, name))
			if err != nil {
				return err
			}
			os.Remove(filepath.Join(thumbnailDir, filepath.Base(mediaFile.LocalThumbnailPath)))
			// err = os.Rename(filepath.Join(thumbnailDir, filepath.Base(mediaFile.LocalThumbnailPath)), filepath.Join(binDir, filepath.Base(mediaFile.LocalThumbnailPath)))
			// if err != nil {
			// 	return err
			// }
			err = bucket.Delete([]byte(name))
			if err != nil {
				return err
			}
		}
		return nil
	})
}

func printDatabaseLength() {
	err := db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte("MediaFiles"))
		if b == nil {
			return nil
		}
		fmt.Println("Database length: ", b.Stats().KeyN)
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

}

func clearDb() {
	err := db.Update(func(tx *bbolt.Tx) error {
		return tx.DeleteBucket([]byte("MediaFiles"))
	})
	if err != nil {
		log.Fatal(err)
	}
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

func isFileallreadyExistsInDb(newFilePath string) (alreadyExists bool, existName string) {
	newChecksum, err := calculateChecksum(newFilePath)
	if err != nil {
		return false, ""
	}
	err = db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte("MediaFiles"))
		if bucket == nil {
			return nil
		}
		return bucket.ForEach(func(k, v []byte) error {
			var mediaFile MediaFile
			if err := json.Unmarshal(v, &mediaFile); err != nil {
				return err
			}
			if mediaFile.Checksum == newChecksum {
				alreadyExists = true
				existName = mediaFile.FileName
				return nil
			}
			return nil
		})
	})

	return alreadyExists, existName
}

func getMediaFiles(page, pageSize int, sortBy, filterBy, sortOrder string) ([]MediaFile, int, error) {
	var mediaFiles []MediaFile
	var totalFiles int

	err := db.View(func(tx *bbolt.Tx) error {

		b := tx.Bucket([]byte("MediaFiles"))
		if b == nil {
			return fmt.Errorf("bucket MediaFiles not found")
		}

		// Collect all media files
		err := b.ForEach(func(k, v []byte) error {
			var mf MediaFile
			if err := json.Unmarshal(v, &mf); err != nil {
				return err
			}
			mediaFiles = append(mediaFiles, mf)
			return nil
		})
		if err != nil {
			return err
		}

		// Apply filtering
		if filterBy != "" {
			mediaFiles = filterMediaFiles(mediaFiles, filterBy)
		}

		// Get total count of files after filtering
		totalFiles = len(mediaFiles)

		// Apply sorting
		sortMediaFiles(mediaFiles, sortBy, sortOrder)

		// Apply pagination
		start := (page - 1) * pageSize
		end := start + pageSize
		if start < len(mediaFiles) {
			if end > len(mediaFiles) {
				mediaFiles = mediaFiles[start:]
			} else {
				mediaFiles = mediaFiles[start:end]
			}
		} else {
			mediaFiles = []MediaFile{}
		}

		return nil
	})

	if err != nil {
		return nil, 0, err
	}

	return mediaFiles, totalFiles, nil
}

func filterMediaFiles(files []MediaFile, filterBy string) []MediaFile {
	var filtered []MediaFile
	filterLower := strings.ToLower(filterBy)
	for _, file := range files {
		if strings.Contains(strings.ToLower(file.FileName), filterLower) {
			filtered = append(filtered, file)
		}
	}
	return filtered
}

func sortMediaFiles(files []MediaFile, sortBy, sortOrder string) {
	sort.Slice(files, func(i, j int) bool {
		var less bool
		switch sortBy {
		case "name":
			less = files[i].FileName < files[j].FileName
		case "date":
			less = files[i].CreationDate.Before(files[j].CreationDate)
		case "size":
			less = files[i].FileSize < files[j].FileSize
		default:
			less = files[i].FileName < files[j].FileName
		}
		if sortOrder == "desc" {
			return !less
		}
		return less
	})
}

func generateUniqueFileName(db *bbolt.DB, originalFileName string) (string, error) {
	// if file extention is .heic - change it to .jpg
	if strings.HasSuffix(originalFileName, ".heic") {
		originalFileName = strings.TrimSuffix(originalFileName, ".heic") + ".jpg"
	}

	var uniqueFileName = originalFileName
	var counter = 0

	for {
		exists, err := fileExists(db, uniqueFileName)
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

func fileExists(db *bbolt.DB, fileName string) (bool, error) {
	var exists bool
	err := db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte("MediaFiles"))
		if bucket == nil {
			// Bucket does not exist, so the file cannot exist
			exists = false
			return nil
		}
		v := bucket.Get([]byte(fileName))
		exists = v != nil
		return nil
	})
	return exists, err
}

func storeMediaFile(db *bbolt.DB, file MediaFile) error {
	return db.Update(func(tx *bbolt.Tx) error {
		b, err := tx.CreateBucketIfNotExists([]byte("MediaFiles"))
		if err != nil {
			return err
		}

		encoded, err := json.Marshal(file)
		if err != nil {
			return err
		}

		return b.Put([]byte(file.FileName), encoded)
	})
}

func testDuplicateInserting() {

	// Example usage
	originalFileName := "sunset.jpg"
	uniqueFileName, err := generateUniqueFileName(db, originalFileName)
	if err != nil {
		log.Fatal(err)
	}

	mediaFile := MediaFile{
		FileName:  uniqueFileName,
		MediaType: "image",
		// ... other fields ...
	}

	err = storeMediaFile(db, mediaFile)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Stored file with unique name: %s\n", uniqueFileName)
}

func printDatabaseMediaFiles() {
	err := db.View(func(tx *bbolt.Tx) error {
		fmt.Println("Database Media Files:")
		fmt.Println("---------------------")
		return tx.ForEach(func(name []byte, b *bbolt.Bucket) error {
			fmt.Printf("Bucket: %s\n", name)
			fmt.Println("---------------------")
			return b.ForEach(func(k, v []byte) error {
				fmt.Printf("  Key: %s, Value: %s\n", k, v)
				return nil
			})
		})
	})

	if err != nil {
		log.Fatal(err)
	}
}

func getMediaFile(db *bbolt.DB, uniqueID string) (*MediaFile, error) {
	var mediaFile MediaFile
	err := db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte("MediaFiles"))
		if bucket == nil {
			return fmt.Errorf("Bucket MediaFiles not found")
		}

		data := bucket.Get([]byte(uniqueID))
		if data == nil {
			return fmt.Errorf("No file found with ID: %s", uniqueID)
		}

		return json.Unmarshal(data, &mediaFile)
	})
	if err != nil {
		return nil, err
	}
	return &mediaFile, nil
}

// type ExifTagNames struct {
// 	ImageWidth            string `json:"imageWidth"`
// 	ImageLength           string `json:"imageLength"`
// 	Make                  string `json:"make"`
// 	Model                 string `json:"model"`
// 	Orientation           string `json:"orientation"`
// 	XResolution           string `json:"xResolution"`
// 	YResolution           string `json:"yResolution"`
// 	ResolutionUnit        string `json:"resolutionUnit"`
// 	Software              string `json:"software"`
// 	DateTime              string `json:"dateTime"`
// 	YCbCrPositioning      string `json:"yCbCrPositioning"`
// 	ExifTag               string `json:"exifTag"`
// 	ExposureTime          string `json:"exposureTime"`
// 	FNumber               string `json:"fNumber"`
// 	ExposureProgram       string `json:"exposureProgram"`
// 	ISOSpeedRatings       string `json:"isoSpeedRatings"`
// 	ExifVersion           string `json:"exifVersion"`
// 	DateTimeOriginal      string `json:"dateTimeOriginal"`
// 	DateTimeDigitized     string `json:"dateTimeDigitized"`
// 	OffsetTime            string `json:"offsetTime"`
// 	OffsetTimeOriginal    string `json:"offsetTimeOriginal"`
// 	ShutterSpeedValue     string `json:"shutterSpeedValue"`
// 	ApertureValue         string `json:"apertureValue"`
// 	BrightnessValue       string `json:"brightnessValue"`
// 	ExposureBiasValue     string `json:"exposureBiasValue"`
// 	MaxApertureValue      string `json:"maxApertureValue"`
// 	MeteringMode          string `json:"meteringMode"`
// 	Flash                 string `json:"flash"`
// 	FocalLength           string `json:"focalLength"`
// 	SubSecTime            string `json:"subSecTime"`
// 	SubSecTimeOriginal    string `json:"subSecTimeOriginal"`
// 	SubSecTimeDigitized   string `json:"subSecTimeDigitized"`
// 	ColorSpace            string `json:"colorSpace"`
// 	PixelXDimension       string `json:"pixelXDimension"`
// 	PixelYDimension       string `json:"pixelYDimension"`
// 	ExposureMode          string `json:"exposureMode"`
// 	WhiteBalance          string `json:"whiteBalance"`
// 	DigitalZoomRatio      string `json:"digitalZoomRatio"`
// 	FocalLengthIn35mmFilm string `json:"focalLengthIn35mmFilm"`
// 	SceneCaptureType      string `json:"sceneCaptureType"`
// 	GPSTag                string `json:"gpsTag"`
// 	GPSLatitudeRef        string `json:"gpsLatitudeRef"`
// 	GPSLatitude           string `json:"gpsLatitude"`
// 	GPSLongitudeRef       string `json:"gpsLongitudeRef"`
// 	GPSLongitude          string `json:"gpsLongitude"`
// 	GPSAltitudeRef        string `json:"gpsAltitudeRef"`
// 	GPSAltitude           string `json:"gpsAltitude"`
// }
