# JoSEFA
## A Job Similarity Evaluator to Forecast submitted HPC job Activities

JoSEFA is a ML workflow which performs a clustering on HPC jobs, and finds similar jobs to 
a new submitted job to compute some predictions.

That repo has been made to run on the Fugaku dataset.

## Licence

The licence is in the LICENCE file applies to the entire repository.

## Install
Create the environment
```bash
conda env create -f environment.yml
conda activate josefa_env
```

## Download Fugaku Dataset
https://zenodo.org/records/11467483

## Launch JoSEFA
```bash
python josefa.py -parquet_dir PARQUET_DIR_PATH -year_month 2023-01 --nb_history_day 30 -retrain_each_days 1
```


## Acknowledgment
This work has been partially supported by European Project HORIZON-EUROHPC-JU-SEANERGYS (g.a. 101177590).