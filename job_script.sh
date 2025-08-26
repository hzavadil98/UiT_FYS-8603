#!/bin/bash


echo "Running the python script..."


#python3 train_synt.py

#python3 train_4v2b.py
python3 --version
#list uv packages

uv --version
ls
uv tree

python3 train_2v1b.py
echo "Script execution completed."