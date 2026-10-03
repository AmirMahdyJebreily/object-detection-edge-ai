#!/bin/bash

CONTAINER_NAME="object-detection-edge-ai-server"

echo "Stopping container: $CONTAINER_NAME..."
docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || echo "Container was not running."

echo "Removing container: $CONTAINER_NAME..."
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || echo "Container was not found."

echo "Done."
