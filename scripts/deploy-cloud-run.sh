#!/bin/bash

# LogSense AI - Cloud Run Deployment Script
# Usage: ./scripts/deploy-cloud-run.sh [PROJECT_ID] [REGION]

set -e

# Configuration
PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"us-central1"}
SERVICE_NAME="logsense-ai"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Deploying LogSense AI to Cloud Run..."
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"

# Verify gcloud authentication
echo "Checking gcloud authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "Error: Please authenticate with gcloud first:"
    echo "   gcloud auth login"
    exit 1
fi

# Set project
echo "Setting gcloud project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push image
echo "Building and pushing Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} .

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --port 8080 \
    --timeout 60s \
    --concurrency 80 \
    --set-env-vars NODE_ENV=production,CLOUD_RUN=true,GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION}

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')

echo "Deployment complete!"
echo "Service URL: ${SERVICE_URL}"
echo "API Docs: ${SERVICE_URL}/api/docs"
echo "Health Check: ${SERVICE_URL}/health"
echo "Metrics: ${SERVICE_URL}/metrics"

# Test deployment
echo "Testing deployment..."
if curl -f -s "${SERVICE_URL}/health" > /dev/null; then
    echo "Health check passed!"
else
    echo "Error: Health check failed!"
    exit 1
fi

echo "LogSense AI is now live at: ${SERVICE_URL}"