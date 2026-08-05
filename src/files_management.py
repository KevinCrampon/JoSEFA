import os
from datetime import datetime

import pandas as pd  # type: ignore


def create_dir(
    base_name: str, use_nb_last_days: str | int, retrain_each_days: str | int
) -> str:
    """Create results directory

    Args:
        base_name (str): The directory base name
        use_nb_last_days (str): The number of days used to train
        retrain_each_days (str): The number of days between each retraining

    Returns:
        str: The directory name
    """
    res_dir = f"{base_name}_{use_nb_last_days}_{retrain_each_days}"
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    return res_dir


def write_global_result_header(file_path: str) -> None:
    """Writes the header for the global results file,
        override the file if it already exist

    Args:
        file_path (str): The file path
    """
    with open(file_path, "w") as f:
        f.writelines(
            [
                "train_start_datetime,train_end_datetime,test_start_datetime,"
                "test_end_datetime,"
                "train_set_size,used_train_set_size,test_set_size,"
                "r2_time,mae_time,mse_time,"
                "acc_pclass,f1_pclass,"
                "clustering_time,min_cluster_size\n"
            ]
        )


def write_global_result_line(
    file_path: str,
    train_start_datetime: datetime,
    train_end_datetime: datetime,
    test_start_datetime: datetime,
    test_end_datetime: datetime,
    nb_base_train_jobs: int,
    nb_actual_train_jobs: int,
    nb_test_jobs: int,
    r2_time: float | None,
    mae_time: float | None,
    mse_time: float | None,
    acc_pclass: float | None,
    f1_pclass: float | None,
    nb_not_predicted: int,
    clustering_time: str,
    min_cluster_size: int,
    date_format: str,
) -> None:
    """Add a new line with all the clustering global result in the file

    Args:
        file_path (str): The file path
        train_start_datetime (datetime): The start datetime for jobs in
        training set
        train_end_datetime (datetime): The end datetime for jobs in training
        set
        test_start_datetime (datetime): The start datetime for jobs in test set
        test_end_datetime (datetime): The end datetime for jobs in test set
        nb_base_train_jobs (int): The number of jobs in the training set before
            subsampling
        nb_actual_train_jobs (int): The number of jobs in the training set
            after subsampling
        nb_test_jobs (int): The number of jobs in the test set
        r2_time (float | None): The R^2 score for duration prediction
        mae_time (float | None): The MAE score for duration prediction
        mse_time (float | None): The MSE score for duration prediction
        acc_pclass (float | None): The Accuracy score for bound prediction
        f1_pclass (float | None): The F1 score for bound prediction
        nb_not_predicted (int): The number of jobs without prediction
        clustering_time (str): The clustering time
        min_cluster_size (int): The min_cluster_size parameter value
        date_format (str): The date format
    """
    r2_time_rounded = round(r2_time, 2) if r2_time is not None else "nan"
    mae_time_rounded = round(mae_time, 2) if mae_time is not None else "nan"
    mse_time_rounded = round(mse_time, 2) if mse_time is not None else "nan"
    acc_pclass_rounded = (
        round(acc_pclass, 2) if acc_pclass is not None else "nan"
    )
    f1_pclass_rounded = round(f1_pclass, 2) if f1_pclass is not None else "nan"
    with open(file_path, "a") as f:
        f.writelines(
            [
                f"{train_start_datetime.strftime(date_format)},"
                f"{train_end_datetime.strftime(date_format)},"
                f"{test_start_datetime.strftime(date_format)},"
                f"{test_end_datetime.strftime(date_format)},"
                f"{nb_base_train_jobs},"
                f"{nb_actual_train_jobs},"
                f"{nb_test_jobs},"
                f"{r2_time_rounded},"
                f"{mae_time_rounded},"
                f"{mse_time_rounded},"
                f"{acc_pclass_rounded},"
                f"{f1_pclass_rounded},"
                f"{nb_not_predicted},"
                f"{clustering_time}"
                f"{min_cluster_size}\n"
            ]
        )


def write_clustering_results(
    file_path: str,
    pred: list[float],
    gt: list[float],
    list_jid: list[str],
    list_date_start: list[str],
    list_date_end: list[str],
) -> None:
    if len(pred) > 0 and len(gt) > 0:
        df_res_pclass_sr = pd.DataFrame(
            {
                "jid": list_jid,
                "pred": pred,
                "gt": gt,
                "date_start": list_date_start,
                "date_end": list_date_end,
            }
        )
        df_res_pclass_sr.to_csv(
            file_path,
            index=False,
        )
