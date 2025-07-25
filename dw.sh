#!/bin/bash

# build-and-push.sh - Docker image build and push script with version incrementing

# Configuration
IMAGE_NAME="worker-whisperx"  # Change this to your image name
DOCKERFILE_PATH="./Dockerfile"  # Path to your Dockerfile
VERSION_FILE="version.txt"      # Path to version file
REGISTRY="docker.io/timberthrax"  # Change to your registry/username

# Function to display usage information
show_usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  --push    Build and push the Docker image"
  echo "  --help    Show this help message"
}

# Parse command arguments
PUSH_IMAGE=false
for arg in "$@"; do
  case $arg in
    --push)
      PUSH_IMAGE=true
      ;;
    --help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      show_usage
      exit 1
      ;;
  esac
done

# Check if version file exists
if [ ! -f "$VERSION_FILE" ]; then
  echo "Error: Version file '$VERSION_FILE' not found."
  echo "Creating it with initial version 0.1.0"
  echo "0.1.0" > "$VERSION_FILE"
fi

# Read current version
VERSION=$(cat "$VERSION_FILE")
echo "Current version: $VERSION"

# Split version into components
IFS='.' read -r MAJOR MINOR PATCH <<< "$VERSION"

# Increment patch version
PATCH=$((PATCH + 1))

# Create new version string
NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
echo "New version: $NEW_VERSION"

# Update version file
echo "$NEW_VERSION" > "$VERSION_FILE"
echo "Updated version file"

# Full image name with tag
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${NEW_VERSION}"
LATEST_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:latest"

# Build Docker image
echo "Building Docker image: $FULL_IMAGE_NAME"
if docker build --build-arg HF_TOKEN="$HF_TOKEN" -t "$FULL_IMAGE_NAME" -f "$DOCKERFILE_PATH" .; then
  echo "Docker image built successfully"
  
  # Tag as latest
  echo "Tagging as latest"
  docker tag "$FULL_IMAGE_NAME" "$LATEST_IMAGE_NAME"
  
  # Push image if requested
  if [ "$PUSH_IMAGE" = true ]; then
    echo "Pushing Docker image: $FULL_IMAGE_NAME"
    if docker push "$FULL_IMAGE_NAME"; then
      echo "Pushed version tag successfully"
      
      echo "Pushing latest tag"
      if docker push "$LATEST_IMAGE_NAME"; then
        echo "Pushed latest tag successfully"
      else
        echo "Failed to push latest tag"
        exit 1
      fi
    else
      echo "Failed to push version tag"
      exit 1
    fi
  else
    echo "Skipping push (use --push to push the image)"
  fi
else
  echo "Docker build failed"
  # Revert version file change on build failure
  echo "$VERSION" > "$VERSION_FILE"
  echo "Reverted version file due to build failure"
  exit 1
fi

echo "Done! New version: $NEW_VERSION"