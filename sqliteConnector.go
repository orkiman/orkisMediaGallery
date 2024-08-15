package main

import (
	"database/sql"
	"fmt"
	"path/filepath"

	_ "github.com/mattn/go-sqlite3" // Import SQLite driver
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
