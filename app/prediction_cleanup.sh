#!/bin/bash

# Define the directory to clean up
PREDICTION_DIR="/usr/src/app/static/data/*"

# Find and delete directories older than 3 minutes
find "$PREDICTION_DIR" -type d -mmin +2 -exec rm -rf {} +

# Define the directory to clean up
IMAGE_DIR="/usr/src/app/images/*"

# Find and delete directories older than 3 minutes
find "$PREDICTION_DIR" -type f -mmin +2 -exec rm -f {} +
