package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/dsoprea/go-exif/v3"
	exifcommon "github.com/dsoprea/go-exif/v3/common"
)

func getOrientation(imagePath string) (int, error) {
	// Read the image file into a byte slice
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		fmt.Println("Error reading image file:", err)
		return -1, err
	}

	// Extract EXIF data
	rawExif, err := exif.SearchAndExtractExif(imageData)
	if err != nil {
		fmt.Println("Error extracting EXIF data:", err)
		return -1, err
	}

	// Parse the EXIF data
	im := exifcommon.NewIfdMapping()
	err = exifcommon.LoadStandardIfds(im)
	if err != nil {
		fmt.Println("Error loading standard IFDs:", err)
		return -1, err
	}

	ti := exif.NewTagIndex()

	_, index, err := exif.Collect(im, ti, rawExif)
	if err != nil {
		fmt.Println("Error collecting EXIF data:", err)
		return -1, err
	}

	// Try to find Orientation tag
	orientationTags, err := index.RootIfd.FindTagWithName("Orientation")
	if err == nil && len(orientationTags) > 0 {
		orientationTag := orientationTags[0]
		value, err := orientationTag.FormatFirst()
		if err == nil {
			// fmt.Printf("Orientation: %v\n", value)
			orientationInt, err := strconv.Atoi(value)
			if err != nil {
				fmt.Println("Error converting orientation to integer:", err)
				return -1, err
			}
			return orientationInt, nil
		}
	}

	return -1, fmt.Errorf("Orientation tag not found")
}

func main2() {
	imagePath := "/home/spot/spot-or/myPhotosTest/photos/7.jpg"
	fmt.Println(getOrientation(imagePath))
}

// 	// Read the image file into a byte slice
// 	imageData, err := os.ReadFile(imagePath)
// 	if err != nil {
// 		fmt.Println("Error reading image file:", err)
// 		return
// 	}

// 	// Extract EXIF data
// 	rawExif, err := exif.SearchAndExtractExif(imageData)
// 	if err != nil {
// 		fmt.Println("Error extracting EXIF data:", err)
// 		return
// 	}

// 	// Parse the EXIF data
// 	im := exifcommon.NewIfdMapping()
// 	err = exifcommon.LoadStandardIfds(im)
// 	if err != nil {
// 		fmt.Println("Error loading standard IFDs:", err)
// 		return
// 	}

// 	ti := exif.NewTagIndex()

// 	_, index, err := exif.Collect(im, ti, rawExif)
// 	if err != nil {
// 		fmt.Println("Error collecting EXIF data:", err)
// 		return
// 	}

// 	// Print all IFDs and their tags
// 	fmt.Println("All IFDs and tags:")
// 	printIfdTags(index.RootIfd)

// 	// Try to find Orientation tag
// 	orientationTags, err := index.RootIfd.FindTagWithName("Orientation")
// 	if err == nil && len(orientationTags) > 0 {
// 		orientationTag := orientationTags[0]
// 		value, err := orientationTag.FormatFirst()
// 		if err == nil {
// 			fmt.Printf("Orientation: %v\n", value)
// 			return
// 		}
// 	}
// 	fmt.Println("Orientation tag not found")
// }

// func printIfdTags(ifd *exif.Ifd) {
// 	if ifd == nil {
// 		return
// 	}

// 	ifdIdentity := ifd.IfdIdentity()
// 	fmt.Printf("IFD: %s\n", ifdIdentity.String())
// 	for _, entry := range ifd.Entries() {
// 		tagId := entry.TagId()
// 		tagType := entry.TagType()
// 		tagName := entry.TagName()
// 		value, _ := entry.FormatFirst()
// 		fmt.Printf("  Tag ID: %d, Type: %d, Name: %s, Value: %v\n", tagId, tagType, tagName, value)
// 	}
// 	fmt.Println()

// 	for _, childIfd := range ifd.Children() {
// 		printIfdTags(childIfd)
// 	}
// }
