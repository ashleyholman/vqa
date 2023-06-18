#!/bin/bash

# Initialize a flag for skipping json data
SKIP_JSON_DATA=0
INVALIDATE_CACHE=0

# Parse arguments
for arg in "$@"
do
    case $arg in
        --skip-json-data)
            SKIP_JSON_DATA=1
            shift
            ;;
        --invalidate-cache)
            INVALIDATE_CACHE=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "--skip-json-data     Skips the contents of build/data/ from the S3 sync step"
            echo "--invalidate-cache   Invalidates the CloudFront cache after syncing to S3"
            echo "--help               Displays this help message"
            exit 0
            ;;
    esac
done

# Change directory to the script's location
cd $( dirname -- "$0"; )

echo Building React app..

# Change to the web-frontend directory and build the react app
cd ../../web-frontend
npm run build

if [ $? -ne 0 ]; then
    echo -e "\nFailed to build the react app. Exiting..."
    exit 1
fi

echo Syncing artefacts to S3..

# If skip json data flag is set, sync excluding json data
if [ $SKIP_JSON_DATA -eq 1 ]; then
    aws s3 sync build/ s3://vqa-web/ --exclude "data/*"
else
    aws s3 sync build/ s3://vqa-web/
fi

if [ $? -ne 0 ]; then
    echo -e "\nFailed to sync the build to the S3 bucket. Exiting..."
    exit 1
fi

# Check if --invalidate-cache option is passed
if [ $INVALIDATE_CACHE -eq 1 ]; then

    echo Retrieving CloudFront distribution ID..

    # Get the CloudFront distribution ID for invalidation
    CLOUDFRONT_DISTRIBUTION_ID=$(aws cloudformation describe-stack-resource --stack-name vqa-stack --logical-resource-id VqaCloudFrontDistribution --query 'StackResourceDetail.PhysicalResourceId' --output text)
    if [ -z "$CLOUDFRONT_DISTRIBUTION_ID" ]; then
        echo -e "\nFailed to get the CloudFront distribution ID. Exiting..."
        exit 1
    fi

    echo Invalidating CloudFront cache..

    # Create CloudFront invalidation
    aws cloudfront create-invalidation --distribution-id $CLOUDFRONT_DISTRIBUTION_ID --paths "/*"
    if [ $? -eq 0 ]; then
        echo -e "\nSuccessfully invalidated CloudFront cache."
    else
        echo -e "\nFailed to invalidate CloudFront cache."
    fi

else
    echo -e "\nCache invalidation skipped.  Use --invalidate-cache option to enable."
fi