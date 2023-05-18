#!/bin/bash
REGION=ap-southeast-1
STACK_NAME="vqa-stack"
EC2_IP_FILE=$( dirname -- "$0"; )/../../EC2_IP_ADDRESS

# Check if the correct number of command line arguments were provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <cpu|gpu>"
  exit 1
fi

# Get instance type from command line argument
INSTANCE_TYPE=$1

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


echo == Describing stack $STACK_NAME..
# Name of your Auto Scaling Group
ASG_NAME=$(aws cloudformation describe-stacks --region $REGION --stack-name $STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='$OUTPUT_KEY'].OutputValue" --output text)
echo Found ASG $ASG_NAME

echo == Retrieving instance ID from ASG..
# Get instance ID
INSTANCE_ID=$(aws autoscaling describe-auto-scaling-groups --region $REGION --auto-scaling-group-name $ASG_NAME --query "AutoScalingGroups[0].Instances[0].InstanceId" --output text)
echo Found instance $INSTANCE_ID

echo == Retrieving instance Public IP
# Get instance public IP
INSTANCE_IP=$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo writing IP $INSTANCE_IP to $EC2_IP_FILE
# Write the IP to the file
echo $INSTANCE_IP > $EC2_IP_FILE
