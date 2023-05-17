#!/bin/bash
STACK_NAME="vqa-stack"
EC2_IP_FILE=$( dirname -- "$0"; )/../../EC2_IP_ADDRESS

echo == Describing stack $STACK_NAME..
# Name of your Auto Scaling Group
ASG_NAME=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='CPUOnlyAutoScalingGroupName'].OutputValue" --output text)
echo Found ASG $ASG_NAME

echo == Retrieving instance ID from ASG..
# Get instance ID
INSTANCE_ID=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-name $ASG_NAME --query "AutoScalingGroups[0].Instances[0].InstanceId" --output text)
echo Found instance $INSTANCE_ID

echo == Retrieving instance Public IP
# Get instance public IP
INSTANCE_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo writing IP $INSTANCE_IP to $EC2_IP_FILE
# Write the IP to the file
echo $INSTANCE_IP > $EC2_IP_FILE
