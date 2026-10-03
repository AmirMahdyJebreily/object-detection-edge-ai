#!/bin/bash
set -e

IMAGE_NAME="object-detection-edge-ai:latest"
CONTAINER_NAME="object-detection-edge-ai-server"

echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

echo "Checking if a previous container is running..."
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Stopping and removing existing container..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

echo "Deploying Docker container: $CONTAINER_NAME"
# Note: --device /dev/video0 is mapped to allow camera access inside the container.
# If your camera device path is different, please change it.
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p 5000:5000 \
    --device /dev/video0:/dev/video0 \
    "$IMAGE_NAME"

echo "Deployment successful! The server should be running on http://localhost:5000"
echo "To view logs, use: docker logs -f $CONTAINER_NAME"
