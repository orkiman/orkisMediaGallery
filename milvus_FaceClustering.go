package main

import (
	"context"
	"log"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	// Milvus instance proxy address, may verify in your env/settings
	milvusAddr = `localhost:19530`

	collectionName                       = `gosdk_basic_collection`
	embeddingDim                         = 512
	mediaIdColName, faceEmbeddingColName = "mediaID", "faceEmbeddings"
)

var milvusClient client.Client

func openMilvus() client.Client {
	// setup context for client creation, use 10 seconds here
	ctx := context.Background()

	var err error
	milvusClient, err = client.NewClient(ctx, client.Config{
		Address: milvusAddr,
	})
	if err != nil {
		// handling error and exit, to make example simple here
		log.Fatal("failed to connect to milvus:", err.Error())
	}

	// Check if the collection exists
	exists, err := milvusClient.HasCollection(ctx, collectionName)
	if err != nil {
		log.Fatalf("Failed to check if collection exists: %v", err)
	}

	if !exists {
		// Create collection

		// define collection schema
		schema3 := entity.NewSchema().WithName(collectionName).WithDescription("face embeddings collection").
			// currently primary key field is compulsory, and only int64 is allowed
			WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
			// the mediaID field
			WithField(entity.NewField().WithName(mediaIdColName).WithDataType(entity.FieldTypeInt64)).
			// also the vector field is needed
			WithField(entity.NewField().WithName(faceEmbeddingColName).WithDataType(entity.FieldTypeFloatVector).WithDim(embeddingDim))

		err = milvusClient.CreateCollection(ctx, schema3, entity.DefaultShardNumber)
		if err != nil {
			log.Fatal("failed to create collection:", err.Error())
		}
	}

	return milvusClient
}

func insertNewEmbeddings(mediaId int64, embedding []float32) {
	// setup context for client creation, use 10 seconds here
	ctx := context.Background()
	milvusClient := openMilvus()

	// prepare data
	mediaIdColData := entity.NewColumnInt64(mediaIdColName, []int64{mediaId})
	embeddingColData := entity.NewColumnFloatVector(faceEmbeddingColName, embeddingDim, [][]float32{embedding})
	// insert data
	if _, err := milvusClient.Insert(ctx, collectionName, "", mediaIdColData, embeddingColData); err != nil {
		log.Fatalf("failed to insert random data into `hello_milvus, err: %v", err)
	}

	if err := milvusClient.Flush(ctx, collectionName, false); err != nil {
		log.Fatalf("failed to flush data, err: %v", err)
	}
}
