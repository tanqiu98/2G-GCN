# Multi-person Human-object Interaction Recognition in Videos
Code for the ECCV'22 paper "Geometric Features Informed Multi-person Human-object Interaction Recognition in Videos".

## Environment Setup
First please create an appropriate environment using conda: 

> conda env create -f environment.yml

> conda activate vhoi

## Download Data
Please download the necessary data from the link below, and put the 
downloaded data folder in this current directory (i.e. `./data/...`).

Link: [data](https://drive.google.com/drive/folders/1yfwItIoQrAnbnk5GTjbbfN8Ls8Ybl_hr?usp=sharing).

## Train a Model
To train a model from scratch, edit the `./conf/config.yaml` file, and depending on the selected dataset and model, also 
edit the associated model .yaml file in `./conf/models/` and the associated dataset .yaml file in `./conf/data/`. After 
editing the files, just run `python train.py`.

