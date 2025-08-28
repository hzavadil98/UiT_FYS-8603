#!/bin/bash


echo "Running the python script..."


#python3 train_synt.py

#python3 train_4v2b.py
apt-get update
apt-get install -y iputils-ping
ping api.wandb.ai
#python3 train_2v1b.py
echo "Script execution completed."