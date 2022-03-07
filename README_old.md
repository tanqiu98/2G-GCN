# Video Human-Object Interaction
Code for the CVPR'21 paper [Learning Asynchronous and Sparse Human-Object Interaction in Videos](https://openaccess.thecvf.com/content/CVPR2021/html/Morais_Learning_Asynchronous_and_Sparse_Human-Object_Interaction_in_Videos_CVPR_2021_paper.html).

## Environment Setup
First please create an appropriate environment using conda: 

> conda env create -f environment.yml

> conda activate vhoi

## Download Data and Pre-Trained Models
Please download the necessary data for the CAD-120 and Bimanual Actions datasets from the link below, and put the 
downloaded data folder in this current directory (i.e. `./data/...`).

Link: [data](https://bit.ly/3s9NWiB).

Pre-trained models can be found in the link below, and the `outputs` folder should be placed in this current 
directory as well (i.e. `./outputs/...`).

Link: [models](https://bit.ly/3jx7tWh).

## Test Pre-Trained Models
Evaluate ASSIGN on CAD-120 dataset:
> python -W ignore predict.py 
>--pretrained_model_dir ./outputs/cad120/assign/hs512_e40_bs16_lr0.001_sc-None_h2h-False_h2o-True_o2h-True_o2o-True_m-v2-v1-att-v3-False-True_sd-0.1-True_os-ind_dn-1-gs_pf-e0s0_c0_sp-0_ihs-False_ios-False_bl-False-1.0-1.0_sl-True-False-4.0-1.0_fl0-0.0_mt-False_pt-True-z_gc0.0_ds3_Subject1 
>--cross_validate

Evaluate ASSIGN on Bimanual Actions dataset:
> python -W ignore predict.py
>--pretrained_model_dir ./outputs/bimanual/assign/hs64_e30_bs32_lr0.001_sc-None_h2h-True_h2o-True_o2h-True_o2o-True_m-v2-v1-att-v3-False-True_sd-0.1-True_os-ind_dn-1-gs_pf-e0s0_c0_sp-0_ihs-False_ios-False_bl-False-1.0-1.0_sl-True-False-4.0-1.0_fl0-0.0_mt-False_pt-True-z_gc0.0_ds3_1 
>--cross_validate

## Train a Model
To train a model from scratch, edit the `./conf/config.yaml` file, and depending on the selected dataset and model, also 
edit the associated model .yaml file in `./conf/models/` and the associated dataset .yaml file in `./conf/data/`. After 
editing the files, just run `python train.py`.

The configuration settings used for the provided pre-trained models can be found inside the pre-trained model 
directory, within the hidden `.hydra` folder. For example, `./outputs/cad120/assign/hs512_e40_bs16_lr0.001_sc-None_h2h-False_h2o-True_o2h-True_o2o-True_m-v2-v1-att-v3-False-True_sd-0.1-True_os-ind_dn-1-gs_pf-e0s0_c0_sp-0_ihs-False_ios-False_bl-False-1.0-1.0_sl-True-False-4.0-1.0_fl0-0.0_mt-False_pt-True-z_gc0.0_ds3_Subject1/.hydra/config.yaml`.
