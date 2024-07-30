package main

// name change test
import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"image"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/disintegration/imaging"
	"github.com/dsoprea/go-exif/v3"
	exifcommon "github.com/dsoprea/go-exif/v3/common"
)

type PageData struct {
	Files       []MediaItem
	Ipv6Address string
	CurrentPage int
	TotalPages  int
	PrevPage    int
	NextPage    int
	Pages       []int
	SortBy      string
	SortOrder   string
	FilterBy    string
}

type processSelectedRequest struct {
	SelectedFiles  []string `json:"selectedFiles"`
	SelectedAction string   `json:"selectedAction"`
}

const rootDir = "/home/orkiman/Pictures/myPhotosTest"
const thumbnailDir = rootDir + "/thumbnails"
const heicDir = rootDir + "/heic"
const uploadDir = rootDir + "/upload"
const mediaDir = rootDir + "/media"
const duplicatesDir = rootDir + "/duplicated"
const binDir = rootDir + "/bin"

const certFile = "/etc/mygoapp/certs/fullchain.pem" //"cert.pem"
const keyFile = "/etc/mygoapp/certs/privkey.pem"    //"key.pem"
// const certFile = "/home/spot/ssl_certs/fullchain.pem" //"cert.pem"
// const keyFile = "/home/spot/ssl_certs/privkey.pem"    //"key.pem"

type Credentials struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

var globalIpv6Address string
var creds Credentials

func init() {
	credFile := filepath.Join(".", "staticFiles/.login_credentials.json")
	file, err := os.Open(credFile)
	if err != nil {
		panic("Could not open credentials file")
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	err = decoder.Decode(&creds)
	if err != nil {
		panic("Could not decode credentials file")
	}
}
func main() {

	// facesMain()
	// return

	var err error
	globalIpv6Address, err = getDefaultIPv6Address()
	fmt.Println(globalIpv6Address)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	os.MkdirAll(rootDir, 0755)
	os.MkdirAll(heicDir, 0755)
	os.MkdirAll(thumbnailDir, 0755)
	os.MkdirAll(uploadDir, 0755)
	os.MkdirAll(mediaDir, 0755)
	os.MkdirAll(duplicatesDir, 0755)
	os.MkdirAll(binDir, 0755)

	db, err := openBboltDb()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer db.Close()

	// deleteAll()
	// fmt.Println("Files deleted successfully")
	// return

	// err = deleteMedia([]string{"20231211_120532.jpg"})
	// if err != nil {
	// 	fmt.Println("Error:", err)
	// }

	err = orginizeNewFiles()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	printDatabaseLength()
	// testCv()
	http.HandleFunc("/", basicAuth(handleRootDirectoryRequests))

	http.Handle("/media/", basicAuth(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.StripPrefix("/media/", http.FileServer(http.Dir(mediaDir))).ServeHTTP(w, r)
	})))

	http.HandleFunc("/thumbnails/", basicAuth(handleThumbnails))

	http.HandleFunc("/processSelected", basicAuth(handleProcessSelected))
	encripted := true
	if encripted {
		// for production:
		fmt.Println("Server starting on port 8443...")
		err = http.ListenAndServeTLS(":8443", certFile, keyFile, nil)
		if err != nil {
			fmt.Println("Failed to start server:", err)
		}
	} else {
		// for localhost testing - no ssl (port 8080):
		port := "80"
		fmt.Printf("Server starting on port %s...\n", port)
		// err = http.ListenAndServe("localhost:8080", nil) // for development access only from localhost
		err = http.ListenAndServe(":"+port, nil) // for development access allow from anywhere
		if err != nil {
			fmt.Println("Failed to start server:", err)
		}
	}

}
func deleteAll() {
	os.RemoveAll(mediaDir)
	os.RemoveAll(thumbnailDir)
	os.RemoveAll(heicDir)
	os.RemoveAll(duplicatesDir)
	os.RemoveAll(binDir)
	clearDb()
}

func handleRootDirectoryRequests(w http.ResponseWriter, r *http.Request) {
	path := filepath.Join(mediaDir, r.URL.Path)
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		http.NotFound(w, r)
		return
	} else if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if info.IsDir() {
		// Parse query parameters
		page := parseQueryParam(r, "page", 1)
		pageSize := parseQueryParam(r, "pageSize", 2000)
		sortBy := r.URL.Query().Get("sortBy")
		if sortBy == "" {
			sortBy = "name"
		}
		sortOrder := r.URL.Query().Get("sortOrder")
		if sortOrder == "" {
			sortOrder = "asc"
		}
		filterBy := r.URL.Query().Get("filterBy")

		// Query the database for files
		mediaItems, totalFiles, err := getMediaItems(page, pageSize, sortBy, filterBy, sortOrder)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Calculate pagination details
		totalPages := (totalFiles + pageSize - 1) / pageSize
		prevPage := page - 1
		if prevPage < 1 {
			prevPage = 1
		}
		nextPage := page + 1
		if nextPage > totalPages {
			nextPage = totalPages
		}

		// Generate page numbers
		pages := generatePageNumbers(page, totalPages)

		// Prepare data for template
		data := PageData{
			Files:       mediaItems,
			Ipv6Address: globalIpv6Address,
			CurrentPage: page,
			TotalPages:  totalPages,
			PrevPage:    prevPage,
			NextPage:    nextPage,
			Pages:       pages,
			SortBy:      sortBy,
			SortOrder:   sortOrder,
			FilterBy:    filterBy,
		}

		// Render template
		tmplFile := "galleryTemplate.html"
		t, err := template.ParseFiles(tmplFile)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		err = t.Execute(w, data)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	} else {

		files, err := os.ReadDir(filepath.Dir(path))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		var currentIndex, totalImages int
		var prevPath, nextPath string

		for i, file := range files {
			// if isImage(file.Name()) {
			if filepath.Base(path) == file.Name() {
				currentIndex = i
			}
			totalImages++
			// }
		}

		if totalImages > 1 {
			prevIndex := (currentIndex - 1 + totalImages) % totalImages
			nextIndex := (currentIndex + 1) % totalImages
			prevPath = filepath.Join(filepath.Dir(r.URL.Path), files[prevIndex].Name())
			nextPath = filepath.Join(filepath.Dir(r.URL.Path), files[nextIndex].Name())
		}

		data := struct {
			ImagePath string
			PrevPath  string
			NextPath  string
		}{
			ImagePath: filepath.Join("media", r.URL.Path),
			PrevPath:  prevPath,
			NextPath:  nextPath,
		}
		var tmplFile string
		if isImage(path) {
			tmplFile = "imageTemplate.html"
		} else {
			tmplFile = "videoTemplate.html"
		}

		t, err := template.ParseFiles(tmplFile)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		err = t.Execute(w, data)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
}

func handleProcessSelected(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req processSelectedRequest
	err := json.NewDecoder(r.Body).Decode(&req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.SelectedAction == "delete" {
		err = deleteMedia(req.SelectedFiles)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	} else {
		http.Error(w, "action not yet implemented", http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusOK)
}

func parseQueryParam(r *http.Request, name string, defaultValue int) int {
	value := r.URL.Query().Get(name)
	if value == "" {
		return defaultValue
	}
	intValue, err := strconv.Atoi(value)
	if err != nil {
		return defaultValue
	}
	return intValue
}

func generatePageNumbers(currentPage, totalPages int) []int {
	var pages []int
	start := currentPage - 2
	end := currentPage + 2

	if start < 1 {
		start = 1
	}
	if end > totalPages {
		end = totalPages
	}

	for i := start; i <= end; i++ {
		pages = append(pages, i)
	}

	return pages
}

func handleThumbnails(w http.ResponseWriter, r *http.Request) {
	thumbnailPath := filepath.Join(thumbnailDir, strings.TrimPrefix(r.URL.Path, "/thumbnails/"))
	http.ServeFile(w, r, thumbnailPath)
}

func basicAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		username, password, ok := r.BasicAuth()
		if !ok || !checkAuth(username, password) {
			w.Header().Set("WWW-Authenticate", `Basic realm="Restricted"`)
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	}
}

func checkAuth(username, password string) bool {
	return username == creds.Username && password == creds.Password
}

func orginizeNewFiles() error {
	numWorkers := 3 // runtime.NumCPU()
	semaphore := make(chan struct{}, numWorkers)
	var wg sync.WaitGroup
	errChan := make(chan error, numWorkers)

	go func() {
		defer close(errChan)
		err := filepath.WalkDir(uploadDir, func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}

			if !entry.IsDir() {
				wg.Add(1)
				go func(path string) {
					defer wg.Done()
					semaphore <- struct{}{}
					defer func() { <-semaphore }()
					if err := processFile(path); err != nil {
						errChan <- err
					}
				}(path)
			}

			return nil
		})

		if err != nil {
			errChan <- err
		}

		wg.Wait()
	}()

	// Collect and return the first error, if any
	for err := range errChan {
		return err
	}
	// printDatabaseMediaItems()
	return nil
}

func processFile(originalFilePath string) error {
	fileExists, exsistingName := isFileallreadyExistsInDb(originalFilePath)
	if fileExists {
		err := os.Rename(originalFilePath, filepath.Join(duplicatesDir,
			filepath.Base(originalFilePath)+"_alreadyExsistsInDataBaseWithTheName_"+exsistingName))
		if err != nil {
			return fmt.Errorf("error moving file: %w", err)
		}
		return nil
	}
	uniqueName, err := generateUniqueFileName(db, filepath.Base(originalFilePath))
	if err != nil {
		return fmt.Errorf("error generating unique file name: %w", err)
	}
	// filepath.Join(filepath.Dir(originalFilePath), uniqueName)

	if isVideoFile(originalFilePath) {
		if err := generateVideoThumbnail(originalFilePath, uniqueName, thumbnailDir); err != nil {
			return fmt.Errorf("error generating thumbnail for video \"%s\": %w", originalFilePath, err)
		}
		if err := os.Rename(originalFilePath, filepath.Join(mediaDir, uniqueName)); err != nil {
			return fmt.Errorf("error moving video file: %w", err)
		}
	} else if isImage(originalFilePath) {
		// tagNames := []string{"DateTimeOriginal"}
		// exifNameValueMap, err := getExifNameValueMap(filePath, tagNames)
		// if err != nil {
		// 	return fmt.Errorf("error extracting exif data: %w", err)
		// }
		// fmt.Println("original date: ", exifNameValueMap["DateTimeOriginal"])
		if filepath.Ext(originalFilePath) == ".heic" {
			if err := convertHeicToJpegAndGenerateThumbnail(originalFilePath, uniqueName, mediaDir, heicDir, thumbnailDir); err != nil {
				return fmt.Errorf("error converting HEIC to JPEG: %w", err)
			}
		} else {
			if err := generatePhotoThumbnail(originalFilePath, uniqueName, thumbnailDir); err != nil {
				return fmt.Errorf("error generating thumbnail for image: %w", err)
			}
			if err := os.Rename(originalFilePath, filepath.Join(mediaDir, uniqueName)); err != nil {
				return fmt.Errorf("error moving image file: %w", err)
			}
		}
	} else {
		return fmt.Errorf("unsupported file type: %s", originalFilePath)
	}

	return insertMediaToDb(filepath.Join(mediaDir, uniqueName))
}

func convertHeicToJpegAndGenerateThumbnail(originalHeicPath string, uniqueName string, mediaDir string, heicDir string, thumbnailDir string) error {
	// dest := strings.TrimSuffix(heicSrcPath, filepath.Ext(heicSrcPath)) + ".jpg"
	newJpgPath := filepath.Join(mediaDir, uniqueName) //strings.TrimSuffix(filepath.Base(originalHeicPath), filepath.Ext(originalHeicPath))+".jpg")
	newHeicPath := filepath.Join(heicDir, filepath.Base(originalHeicPath))
	// cmd := exec.Command("magick", "convert", originalHeicPath, newJpgPath)
	cmd := exec.Command("convert", originalHeicPath, newJpgPath)
	err := cmd.Run()
	if err != nil {
		return err
	}

	os.Rename(originalHeicPath, newHeicPath)
	// os.Rename(dest, newJpgPath)
	return generatePhotoThumbnail(newJpgPath, uniqueName, thumbnailDir)
}

func generatePhotoThumbnail(originalImagePath string, uniqueName string, thumbnailDir string) error {
	// /* if thumbnail allready exists return	 */
	// if _, err := os.Stat(filepath.Join(thumbnailDir, strings.TrimSuffix(filepath.Base(originalImagePath), filepath.Ext(originalImagePath))+"_thumb.jpg")); err == nil {
	// 	return nil
	// }

	img, err := imaging.Open(originalImagePath)
	if err != nil {
		return err
	}

	orientation, err := getOrientation(originalImagePath)
	// if err != nil {
	// 	return err
	// }
	if err == nil {
		img = adjustOrientation(img, orientation)
	}

	thumbnail := imaging.Resize(img, 200, 0, imaging.Lanczos)
	newThmbnailPath := filepath.Join(thumbnailDir, strings.TrimSuffix(filepath.Base(uniqueName), filepath.Ext(originalImagePath))+"_thumb.jpg")

	return imaging.Save(thumbnail, newThmbnailPath)
}

// New function to generate a thumbnail for a video
func generateVideoThumbnail(videoPath string, uniqueName string, thumbnailDir string) error {
	thumbnailPath := filepath.Join(thumbnailDir, strings.TrimSuffix(filepath.Base(uniqueName), filepath.Ext(uniqueName))+"_thumb.jpg")
	duration, err := getVideoDuration(videoPath)
	if err != nil {
		fmt.Printf("Error getting video duration: %v\n", err)
		return err
	}

	var timestamp float64
	if duration > 4 {
		timestamp = 2
	} else {
		timestamp = duration / 2
	}
	var stderr bytes.Buffer
	cmd := exec.Command("ffmpeg", "-i", videoPath, "-ss", fmt.Sprintf("%f", timestamp), "-vframes", "1", "-vf", "scale=200:-1", thumbnailPath)
	cmd.Stderr = &stderr // Capture standard error
	err = cmd.Run()
	if err != nil {
		log.Printf("FFmpeg error: %s", stderr.String()) // Log the error message from FFmpeg
		return err
	}
	return nil
}
func getVideoDuration(filePath string) (float64, error) {
	cmd := exec.Command("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filePath)
	var out bytes.Buffer
	cmd.Stdout = &out
	err := cmd.Run()
	if err != nil {
		return 0, err
	}
	durationStr := strings.TrimSpace(out.String())
	duration, err := strconv.ParseFloat(durationStr, 64)
	if err != nil {
		return 0, err
	}
	return duration, nil
}

func adjustOrientation(img image.Image, orientation int) image.Image {
	switch orientation {
	case 2:
		return imaging.FlipH(img)
	case 3:
		return imaging.Rotate180(img)
	case 4:
		return imaging.FlipV(img)
	case 5:
		return imaging.Transpose(img)
	case 6:
		return imaging.Rotate270(img)
	case 7:
		return imaging.Transverse(img)
	case 8:
		return imaging.Rotate90(img)
	default:
		return img
	}
}

func getOrientation(imagePath string) (int, error) {
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		return -1, err
	}

	rawExif, err := exif.SearchAndExtractExif(imageData)
	if err != nil {
		return -1, err
	}

	im := exifcommon.NewIfdMapping()
	err = exifcommon.LoadStandardIfds(im)
	if err != nil {
		return -1, err
	}

	ti := exif.NewTagIndex()

	_, index, err := exif.Collect(im, ti, rawExif)
	if err != nil {
		return -1, err
	}

	orientationTags, err := index.RootIfd.FindTagWithName("Orientation")
	if err == nil && len(orientationTags) > 0 {
		orientationTag := orientationTags[0]
		value, err := orientationTag.FormatFirst()
		if err == nil {
			orientationInt, err := strconv.Atoi(value)
			if err != nil {
				return -1, err
			}
			return orientationInt, nil
		}
	}

	return -1, fmt.Errorf("orientation tag not found")
}

// func getExifDataByTagName(imagePath string, tagName string) (string, error) {
// 	// this needs more work...
// 	imageData, err := os.ReadFile(imagePath)
// 	if err != nil {
// 		return "", err
// 	}

// 	rawExif, err := exif.SearchAndExtractExif(imageData)
// 	if err != nil {
// 		return "", err
// 	}

// 	im := exifcommon.NewIfdMapping()
// 	err = exifcommon.LoadStandardIfds(im)
// 	if err != nil {
// 		return "", err
// 	}

// 	ti := exif.NewTagIndex()

// 	_, index, err := exif.Collect(im, ti, rawExif)
// 	if err != nil {
// 		return "", err
// 	}

// 	// fmt.Println(index.RootIfd.EnumerateTagsRecursively())
// 	// tagEntry, err := index.RootIfd.FindTagWithName(tagName)
// 	tagEntry, err := index.RootIfd.FindTagWithName(tagName)
// 	// tagEntry, err := index.Ifds[1].FindTagWithName("DateTimeOriginal")

// 	if err == nil && len(tagEntry) > 0 {
// 		tagValue := tagEntry[0]
// 		value, err := tagValue.Format()
// 		if err == nil {

// 			return value, nil
// 		}
// 	}

// 	return "", err //fmt.Errorf("%s tag not found", tagName)
// }

func isImage(filename string) bool {
	ext := strings.ToLower(filepath.Ext(filename))
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".gif" || ext == ".heic"
}

func isVideoFile(filePath string) bool {
	videoExtensions := map[string]bool{
		".mp4": true,
		".avi": true,
		".mov": true,
		// Add other video formats as needed
	}
	ext := strings.ToLower(filepath.Ext(filePath))
	return videoExtensions[ext]
}

func printAllExifTags(filePath string) error {
	f, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	// how to read file as byte array
	file, err := os.Open(filePath)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer file.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return err
	}

	rawExif, err := exif.SearchAndExtractExif(data)
	if err != nil {
		if err == exif.ErrNoExif {
			fmt.Println("No EXIF data.")
			return nil
		}
		return err
	}

	entries, _, err := exif.GetFlatExifDataUniversalSearch(rawExif, nil, true)
	// entries, _, err := exif.GetFlatExifData(rawExif, nil)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		fmt.Printf("IFD-PATH=[%s] ID=(0x%04x) NAME=[%s] COUNT=(%d) TYPE=[%s] VALUE=[%s]\n", entry.IfdPath, entry.TagId, entry.TagName, entry.UnitCount, entry.TagTypeName, entry.Formatted)
		fmt.Println(entry.TagName, " ", entry.Value)
	}

	return nil
}

// func getAllExifEntries(filePath string, tagNames []string) ([]string, error) {
// 	f, err := os.Open(filePath)
// 	if err != nil {
// 		return nil, err
// 	}
// 	defer f.Close()

// 	// how to read file as byte array
// 	file, err := os.Open(filePath)
// 	if err != nil {
// 		fmt.Println(err)
// 		os.Exit(1)
// 	}
// 	defer file.Close()

// 	data, err := io.ReadAll(f)
// 	if err != nil {
// 		return nil, err
// 	}

// 	rawExif, err := exif.SearchAndExtractExif(data)
// 	if err != nil {
// 		return nil, err
// 	}

// 	entries, _, err := exif.GetFlatExifDataUniversalSearch(rawExif, nil, true)
// 	// entries, _, err := exif.GetFlatExifData(rawExif, nil)
// 	if err != nil {
// 		return nil, err
// 	}
// 	return entries, nil

// }

func getExifNameValueMap(filePath string, tagNames []string) (map[string]string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	rawExif, err := exif.SearchAndExtractExif(data)
	if err != nil {
		if err == exif.ErrNoExif {
			fmt.Println("No EXIF data.")
			return nil, nil
		}
		return nil, err
	}

	entries, _, err := exif.GetFlatExifDataUniversalSearch(rawExif, nil, true)
	if err != nil {
		return nil, err
	}

	result := make(map[string]string)
	for _, tagName := range tagNames {
		for _, entry := range entries {
			if entry.TagName == tagName {
				result[tagName] = entry.Formatted
				break
			}
		}
	}

	return result, nil
}
