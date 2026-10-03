#!/bin/bash

IMAGE_NAME="object-detection-edge-ai:latest"

echo "Removing Docker image: $IMAGE_NAME..."
docker rmi "$IMAGE_NAME" || echo "Image was not found or could not be removed."

echo "Done."
