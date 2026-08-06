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
usage: josefa.py [-h] [-parquet_dir PARQUET_DIR] [-year_month YEAR_MONTH] [-nb_history_days NB_HISTORY_DAYS] [-retrain_each_days RETRAIN_EACH_DAYS] [-strategy {as,sr,su,sur}]

Process some integers.

options:
  -h, --help            show this help message and exit
  -parquet_dir PARQUET_DIR
                        Path to the parquet files directory
  -year_month YEAR_MONTH
                        format: YYYY-MM
  -nb_history_days NB_HISTORY_DAYS, -n NB_HISTORY_DAYS
                        The number of history days used to train the model
  -retrain_each_days RETRAIN_EACH_DAYS, -r RETRAIN_EACH_DAYS
                        Retrain each nb days, is also the number of days used to test the model
  -strategy {as,sr,su,sur}, -s {as,sr,su,sur}
                        The strategy used to perform prediction
```

**Available strategies are:**
* **as**: All similar jobs are used to perform prediction
* **sr**: Only similar jobs with same requested resources are used to perform prediction
* **su**: Only jobs from the same user are used to perform prediction
* **sur**: Only jobs with the same requested resources AND from the same user are used to perform prediction

**Example**
```bash
python josefa.py -parquet_dir PARQUET_DIR_PATH -year_month 2023-01 -nb_history_days 30 -retrain_each_days 1 -s as
```

## Contribution

### Formatting

```bash
black --line-length 79 josefa.py src/*.py
isort --profile black --line-length 79 josefa.py src/*.py
```

## Acknowledgment
The SEANERGYS project receives funding from the European High-Performance Computing Joint Undertaking (JU) under grant agreement no 101177590.

The JU receives support from the European Union's Horizon Europe research and innovation programme and Czech Republic, France, Germany, Greece, Italy, and Spain.
