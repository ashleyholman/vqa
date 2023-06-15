#!/bin/bash

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

# Sync the build to S3 bucket
aws s3 sync build/ s3://vqa-web/
if [ $? -ne 0 ]; then
    echo -e "\nFailed to sync the build to the S3 bucket. Exiting..."
    exit 1
fi

# Check if --invalidate-cache option is passed
if [ "$1" == "--invalidate-cache" ]; then

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
