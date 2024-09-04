package main

// name change test
import (
	"bufio"
	"bytes"
	"database/sql"
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

var db *sql.DB

func main() {
	// deleteAll()
	// fmt.Println("Files deleted successfully")
	// return

	log.SetFlags(log.LstdFlags | log.Lshortfile)
	var err error

	db, err = sql.Open("sqlite3", "./orkisMediaGallery.db")
	if err != nil {
		log.Fatal(err)
		return
	}
	defer db.Close()
	err = createMediaItemsTable(db)
	if err != nil {
		log.Fatal(err)
		return
	}

	// testViewFaces(db)
	// return

	// facesMain()
	// return

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

	processNewFilesInBkgrnd()

	// testCv()
	// http.HandleFunc("/", basicAuth(handleRootDirectoryRequests))
	http.Handle("/", basicAuth(http.HandlerFunc(handleRootDirectoryRequests)))
	http.Handle("/media/", basicAuth(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.StripPrefix("/media/", http.FileServer(http.Dir(mediaDir))).ServeHTTP(w, r)
	})))

	http.HandleFunc("/thumbnails/", basicAuth(handleThumbnails))

	http.HandleFunc("/processSelected", basicAuth(handleProcessSelected))
	http.HandleFunc("/facesTemplate", basicAuth(processFacesTemplate))

	encripted := true
	if encripted {
		// for production:
		fmt.Println("Server starting on port 8443...")
		err = http.ListenAndServeTLS(":8443", certFile, keyFile, nil)
		fmt.Println("after ListenAndServeTLS")
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
	os.RemoveAll("media.db")
	os.RemoveAll("faces.db")
	os.RemoveAll("orkisMediaGallery.db")

	// clearDb()
}

func handleRootDirectoryRequests(w http.ResponseWriter, r *http.Request) {
	smiley := r.URL.Path
	if smiley == "/smiley.png" {
		http.ServeFile(w, r, "smiley.png")
		return
	}
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
			sortOrder = "ASC"
		}
		filterBy := r.URL.Query().Get("filterBy")

		personID := r.URL.Query().Get("personID")
		// log.Print("page:", page, "pageSize:", pageSize, "sortBy:", sortBy, "sortOrder:", sortOrder, "filterBy:", filterBy, "personID:", personID)

		// Query the database for files
		mediaItems, totalFiles, err := getMediaItemsFilteredAndSorted(db, page, pageSize, sortBy, filterBy, sortOrder, personID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// // testing clustering:
		// mediaItems = getbiggestClusterGroup()
		// totalFiles = len(mediaItems)

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

func processFacesTemplate(w http.ResponseWriter, r *http.Request) {

	faces, err := getOneImagePerPersonWithoutPersonsTable(db)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if faces == nil {
		http.Error(w, "No faces found", http.StatusNotFound)
		return
	}
	t, err := template.ParseFiles("facesTemplate.html")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	err = t.Execute(w, faces)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
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
		log.Print("todo: fix delete selected files  - or reclustering logic yet not implemented\n")
		return
		for _, mediaIDsAsString := range req.SelectedFiles {
			mediaID, err := strconv.Atoi(mediaIDsAsString)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			err = deleteMedia(db, mediaID)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
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

func orginizeNewFiles() (processedFilesCounter int, err error) {
	// embeddingsDb = openEmbeddingsDb()
	// defer embeddingsDb.Close()

	err = createFacesUnprocessedMediaItemsTable(db)
	if err != nil {
		panic(fmt.Sprintf("failed to create facesUnprocessedMediaItems table: %v", err))
	}

	err = filepath.WalkDir(uploadDir, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if !entry.IsDir() {
			if err := processFile(path); err != nil {
				log.Printf("Error processing file %s: %v\n", path, err)
				// Continue with the next file instead of returning the error
				return nil
			}
			processedFilesCounter++
		}

		return nil
	})

	if err != nil {
		return processedFilesCounter, err
	}

	// printDatabaseMediaItems()
	return processedFilesCounter, nil
}

func processFile(filePath string) error {
	// if file heic - convert it to jpg first
	if filepath.Ext(filePath) == ".heic" {
		originalHeicPath := filePath
		var err error
		filePath, err = convertHeicToJpg(filePath, uploadDir)
		if err != nil {
			log.Println(err)
			return fmt.Errorf("error converting HEIC to JPEG: %w", err)
		}
		// move heic to outer folder
		newHeicPath := filepath.Join(heicDir, filepath.Base(originalHeicPath))
		err = os.Rename(originalHeicPath, newHeicPath)
		if err != nil {
			return fmt.Errorf("error moving HEIC file: %w", err)
		}
	}
	fileExists, exsistingName := isFileAllreadyExistsInSqlDb(db, filePath)
	if fileExists {
		err := os.Rename(filePath, filepath.Join(duplicatesDir,
			filepath.Base(filePath)+"_alreadyExsistsInDataBaseWithTheName_"+exsistingName))
		if err != nil {
			return fmt.Errorf("error moving file: %w", err)
		}
		return nil
	}
	uniqueName, err := generateUniqueFileName(db, filepath.Base(filePath))

	if err != nil {
		return fmt.Errorf("error generating unique file name: %w", err)
	}
	// filepath.Join(filepath.Dir(originalFilePath), uniqueName)

	if isVideoFile(filePath) {
		if err := generateVideoThumbnail(filePath, uniqueName, thumbnailDir); err != nil {
			return fmt.Errorf("error generating thumbnail for video \"%s\": %w", filePath, err)
		}
		if err := os.Rename(filePath, filepath.Join(mediaDir, uniqueName)); err != nil {
			return fmt.Errorf("error moving video file: %w", err)
		}
	} else if isImage(filePath) {
		if err := generatePhotoThumbnail(filePath, uniqueName, thumbnailDir); err != nil {
			log.Println(err)
			return fmt.Errorf("error generating thumbnail for image: %w", err)
		}
		if err := os.Rename(filePath, filepath.Join(mediaDir, uniqueName)); err != nil {
			return fmt.Errorf("error moving image file: %w", err)
		}

	} else {
		return fmt.Errorf("unsupported file type: %s", filePath)
	}

	mediaItem, err := insertNewMediaToSqlDbAndGetNewMediaItem(db, filepath.Join(mediaDir, uniqueName))
	if err != nil {
		log.Panic(err)
		return err
	}
	// _, err = insertMediaToDb(filepath.Join(mediaDir, uniqueName))
	err = insertMediaToFacesUnprocessedMediaItemsTable(db, &mediaItem)
	if err != nil {
		log.Panic(err)
	}
	return err
}
func convertHeicToJpg(orgininalFilePath, targetDir string) (jpgPath string, err error) {

	jpgName := strings.TrimSuffix(filepath.Base(orgininalFilePath), filepath.Ext(orgininalFilePath)) + ".jpg"
	jpgName, err = getUniqueFileName(jpgName, targetDir)
	if err != nil {
		return "", err
	}
	jpgPath = filepath.Join(targetDir, jpgName)
	cmd := exec.Command("convert", orgininalFilePath, jpgPath)
	err = cmd.Run()
	if err != nil {
		return "", err
	}
	return jpgPath, nil
}

func getUniqueFileName(fileName string, dir string) (string, error) {
	var uniqueName = fileName
	var counter = 0

	for {
		_, err := os.Stat(filepath.Join(dir, uniqueName))
		if err != nil {
			if os.IsNotExist(err) {
				return uniqueName, nil
			}
			return "", err
		}
		counter++
		ext := filepath.Ext(fileName)
		baseName := strings.TrimSuffix(fileName, ext)
		uniqueName = fmt.Sprintf("%s(%d)%s", baseName, counter, ext)
	}
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

// func doPythonClustering() {
// 	log.Println("Python clustering script starting")

// 	cmd := exec.Command("/home/orkiman/vscodeProjects/go/orkisMediaGallery/myDeepFaceVenv/bin/python3", "face_grouping.py")

// 	if err := cmd.Start(); err != nil {
// 		log.Fatal(err)
// 	}

// 	// Do other work in Go while the script runs

// 	go func() { // Wait for the script to finish
// 		err := cmd.Wait()
// 		log.Println("Python clustering script finished")
// 		if err != nil {
// 			log.Fatalf("Python script finished with error: %v", err)
// 		}
// 	}()

// }

func doPythonClustering() {
	log.Println("Python clustering script starting")

	cmd := exec.Command("/home/orkiman/vscodeProjects/go/orkisMediaGallery/myDeepFaceVenv/bin/python3", "face_grouping.py")

	// Create pipes for stdout and stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Fatal(err)
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}

	// Read from stdout and stderr in separate goroutines
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			log.Println("Python stdout:", scanner.Text())
		}
	}()

	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			log.Println("Python stderr:", scanner.Text())
		}
	}()

	// Wait for the command to finish in a separate goroutine
	go func() {
		err := cmd.Wait()
		if err != nil {
			log.Printf("Python script finished with error: %v", err)
		} else {
			log.Println("Python clustering script finished successfully")
		}
	}()

}
func processNewFilesInBkgrnd() {
	go func() {
		processedFilesCounter, err := orginizeNewFiles()
		if err != nil {
			fmt.Println("Error:", err)
			return
		}

		if processedFilesCounter > 0 || true { // do clustring anyway
			// run clustering
			doPythonClustering()
		}

		printDatabaseLength(db)

	}()

}
