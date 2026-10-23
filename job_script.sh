#!/bin/bash


echo "Running the python script..."


#python3 train_synt.py

#python3 train_4v2b.py

#python3 train_2v1b.py

#python3 -u train_1vf.py

#python3 -u train_heads.py

#python3 -u train_elements.py

#python3 train_1vf.py --task 1 --cancer-label-type binary

python3 train_heads.py --cancer-ckpt artifacts/model-r3u0rezv:v0/model.ckpt \
    --feature-dir saved_features/One_View_Canc_vs_Dens_binary --cancer-label-type binary

echo "Script execution completed."