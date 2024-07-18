#!/bin/bash

# Directory to search for HEIC and JPG files
TARGET_DIR="/home/spot/spot-or/myPhotosTest"

# Find all HEIC files in the target directory
find "$TARGET_DIR" -type f -iname "*.heic" | while read heic_file; do
    # Get the base name without extension
    base_name="${heic_file%.*}"
    # Construct the corresponding JPEG file name
    jpg_file="${base_name}.jpg"
    # Check if the JPEG file exists
    if [ -f "$jpg_file" ]; then
        # Delete the JPEG file
        echo "Deleting: $jpg_file"
        rm "$jpg_file"
    fi
done
