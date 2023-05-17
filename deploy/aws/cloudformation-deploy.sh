#!/bin/bash

# Change directory to the deploy dir
cd $( dirname -- "$0"; )

STACK_NAME="vqa-stack"
TEMPLATE_FILE="template.yaml"
REGION="us-east-1"

aws cloudformation deploy \
--stack-name $STACK_NAME \
--template-file $TEMPLATE_FILE \
--region $REGION \
--capabilities CAPABILITY_NAMED_IAM

if [ $? -eq 0 ]; then
    echo -e "\nStack update initiated, waiting for update to complete..."
    aws cloudformation wait stack-update-complete --region $REGION --stack-name $STACK_NAME
    if [ $? -eq 0 ]; then
        echo -e "\nStack update completed successfully."
    else
        echo -e "\nStack update failed. Check the AWS Management Console for details."
    fi
else
    echo -e "\nFailed to initiate stack update."
fi
