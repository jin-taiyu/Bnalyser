#!/bin/bash
#BUSB -J Bnalyser
#BSUB -q normal
#BUSB -n 4
#BUSB -R span[ptile=4]
#BSUB -o train/log/cla_out.txt
#BSUB -e train/log/cla_error.txt
#BSUB -gpu  "num=2"

export LD_LIBRARY_PATH=$HOME/anaconda3/lib:$LD_LIBRARY_PATH

cd '/share/home/Nau222090218/Projects/Bnalyser/train'
python trainCla.py