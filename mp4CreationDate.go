package main

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"time"
)

func getvideoCreationTime(filePath string) (time.Time, error) {
	cmd := exec.Command("ffprobe",
		"-v", "quiet",
		"-print_format", "json",
		"-show_entries", "format_tags=creation_time",
		"-i", filePath)

	output, err := cmd.Output()
	if err != nil {
		return time.Time{}, fmt.Errorf("ffprobe command failed: %v", err)
	}

	var result struct {
		Format struct {
			Tags struct {
				CreationTime string `json:"creation_time"`
			} `json:"tags"`
		} `json:"format"`
	}

	err = json.Unmarshal(output, &result)
	if err != nil {
		return time.Time{}, fmt.Errorf("failed to parse JSON output: %v", err)
	}

	if result.Format.Tags.CreationTime == "" {
		return time.Time{}, fmt.Errorf("creation time not found in file metadata")
	}

	creationTime, err := time.Parse(time.RFC3339Nano, result.Format.Tags.CreationTime)
	if err != nil {
		return time.Time{}, fmt.Errorf("failed to parse creation time: %v", err)
	}

	// fmt.Printf("Creation time: %v\n", creationTime)
	return creationTime, nil
}
