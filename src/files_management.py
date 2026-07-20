from datetime import datetime
import os

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
                "r2_time_same_resources,mae_time_same_resources,"
                "mse_time_same_resources,"
                "acc_pclass_same_resources,f1_pclass_same_resources,"
                "r2_time_all_similar,mae_time_all_similar,mse_time_all_similar,"
                "acc_pclass_all_similar,f1_pclass_all_similar,"
                "nb_not_predicted_same_resources,nb_not_predicted_all_similar,"
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
    r2_time_same_resources: float | None,
    mae_time_same_resources: float | None,
    mse_time_same_resources: float | None,
    acc_pclass_same_resources: float | None,
    f1_pclass_same_resources: float | None,
    r2_time_all_similar: float | None,
    mae_time_all_similar: float | None,
    mse_time_all_similar: float | None,
    acc_pclass_all_similar: float | None,
    f1_pclass_all_similar: float | None,
    nb_not_predicted_same_resources: int,
    nb_not_predicted_all_similar: int,
    clustering_time: str,
    min_cluster_size: int,
    date_format: str,
) -> None:
    """Add a new line with all the clustering global result in the file

    Args:
        file_path (str): The file path
        train_start_datetime (datetime): The start datetime for jobs in training set
        train_end_datetime (datetime): The end datetime for jobs in training set
        test_start_datetime (datetime): The start datetime for jobs in test set
        test_end_datetime (datetime): The end datetime for jobs in test set
        nb_base_train_jobs (int): The number of jobs in the training set before
            subsampling
        nb_actual_train_jobs (int): The number of jobs in the training set after
            subsampling
        nb_test_jobs (int): The number of jobs in the test set
        r2_time_same_resources (float | None): The R^2 score for same resources
            strategy, for duration prediction
        mae_time_same_resources (float | None): The MAE score for same resources
            strategy, for duration prediction
        mse_time_same_resources (float | None): The MSE score for same resources
            strategy, for duration prediction
        acc_pclass_same_resources (float | None): The Accuracy score for same resources
            strategy, for bound prediction
        f1_pclass_same_resources (float | None): The F1 score for same resources
            strategy, for bound prediction
        r2_time_all_similar (float | None): The R^2 score for all similar jobs strategy,
            for duration prediction
        mae_time_all_similar (float | None): The MAE score for all similar jobs
            strategy, for duration prediction
        mse_time_all_similar (float | None): The MSE score for all similar jobs
            strategy, for duration prediction
        acc_pclass_all_similar (float | None): The Accuracy score for all similar jobs
            strategy, for bound prediction
        f1_pclass_all_similar (float | None): The F1 score for all similar jobs
            strategy, for bound prediction
        nb_not_predicted_same_resources (int): The number of jobs without prediction
            for the same resources strategy
        nb_not_predicted_all_similar (int): The number of jobs without prediction for
            the all similar jobs strategy
        clustering_time (str): The clustering time
        min_cluster_size (int): The min_cluster_size parameter value
        date_format (str): The date format
    """
    r2_time_same_resources_rounded = (
        round(r2_time_same_resources, 2)
        if r2_time_same_resources is not None
        else "nan"
    )
    mae_time_same_resources_rounded = (
        round(mae_time_same_resources, 2)
        if mae_time_same_resources is not None
        else "nan"
    )
    mse_time_same_resources_rounded = (
        round(mse_time_same_resources, 2)
        if mse_time_same_resources is not None
        else "nan"
    )
    acc_pclass_same_resources_rounded = (
        round(acc_pclass_same_resources, 2)
        if acc_pclass_same_resources is not None
        else "nan"
    )
    f1_pclass_same_resources_rounded = (
        round(f1_pclass_same_resources, 2)
        if f1_pclass_same_resources is not None
        else "nan"
    )
    r2_time_all_similar_rounded = (
        round(r2_time_all_similar, 2) if r2_time_all_similar is not None else "nan"
    )
    mae_time_all_similar_rounded = (
        round(mae_time_all_similar, 2) if mae_time_all_similar is not None else "nan"
    )
    mse_time_all_similar_rounded = (
        round(mse_time_all_similar, 2) if mse_time_all_similar is not None else "nan"
    )
    acc_pclass_all_similar_rounded = (
        round(acc_pclass_all_similar, 2)
        if acc_pclass_all_similar is not None
        else "nan"
    )
    f1_pclass_all_similar_rounded = (
        round(f1_pclass_all_similar, 2) if f1_pclass_all_similar is not None else "nan"
    )
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
                f"{r2_time_same_resources_rounded},"
                f"{mae_time_same_resources_rounded},"
                f"{mse_time_same_resources_rounded},"
                f"{acc_pclass_same_resources_rounded},"
                f"{f1_pclass_same_resources_rounded},"
                f"{r2_time_all_similar_rounded},"
                f"{mae_time_all_similar_rounded},"
                f"{mse_time_all_similar_rounded},"
                f"{acc_pclass_all_similar_rounded},"
                f"{f1_pclass_all_similar_rounded},"
                f"{nb_not_predicted_same_resources},"
                f"{nb_not_predicted_all_similar},"
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
