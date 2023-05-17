#!/bin/bash

# Specify your region
REGION="us-east-1"

# Specify the name of your CloudFormation stack
STACK_NAME="vqa-stack"

# Get the name of your Auto Scaling group from the CloudFormation stack
ASG_NAME=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='CPUOnlyAutoScalingGroupName'].OutputValue" --output text)

# Check if a command line argument was provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <desired_capacity>"
  exit 1
fi

# Set desired capacity from command line argument
DESIRED_CAPACITY=$1

# Set minimum and maximum size
MIN_SIZE=0
MAX_SIZE=1

# Update the Auto Scaling group
aws autoscaling update-auto-scaling-group --auto-scaling-group-name $ASG_NAME --desired-capacity $DESIRED_CAPACITY --min-size $MIN_SIZE --max-size $MAX_SIZE --region $REGION

# Check the return code of the previous command
if [ $? -eq 0 ]; then
  echo "Successfully updated Auto Scaling group $ASG_NAME to desired capacity $DESIRED_CAPACITY."
else
  echo "Failed to update Auto Scaling group $ASG_NAME. Please check your AWS CLI configuration and network connectivity."
  exit 1
fi
