#!/bin/bash

# AWS region
REGION="us-east-1"

# Cloudformation stack name
STACK_NAME="vqa-stack"

# Check if the correct number of command line arguments were provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <cpu|gpu> <desired_capacity>"
  exit 1
fi

# Get instance type and desired capacity from command line arguments
INSTANCE_TYPE=$1
DESIRED_CAPACITY=$2

# Check if the instance type argument is valid
if [ "$INSTANCE_TYPE" != "cpu" ] && [ "$INSTANCE_TYPE" != "gpu" ]; then
  echo "Invalid instance type. Must be 'cpu' or 'gpu'."
  exit 1
fi

# Determine the cloudformation output key that will provide the ASG name based on the instance type
if [ "$INSTANCE_TYPE" == "cpu" ]; then
  OUTPUT_KEY="CPUOnlyAutoScalingGroupName"
else
  OUTPUT_KEY="GPUAutoScalingGroupName"
fi

# Get the name of the Auto Scaling group from the CloudFormation stack
ASG_NAME=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='$OUTPUT_KEY'].OutputValue" --output text)

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
