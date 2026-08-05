import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore

from src.constants import LIST_MONTH


def load_df(parquet_dir: str, month_year: str) -> pd.DataFrame:
    """Load

    Args:
        parquet_dir (str): Directory where are located all the parquet files
        month_year (str): The parquet name format YY_MM

    Returns:
        pd.DataFrame: A dataframe containing the month_year parquet and also
            the month before and after
    """
    index = LIST_MONTH.index(month_year)
    used_months_year = [month_year]
    if index > 0:
        used_months_year += [LIST_MONTH[index - 1]]
    if index < len(LIST_MONTH) - 1:
        used_months_year += [LIST_MONTH[index + 1]]
    list_df = []
    list_paths = [
        os.path.join(parquet_dir, f"{my}.parquet") for my in used_months_year
    ]
    print("Files to load : ", list_paths)
    for p_file in list_paths:
        p_file = str(Path(p_file).resolve())
        df = pd.read_parquet(p_file)
        df = df[df["exit state"] == "completed"]
        df = df[
            [
                "jid",
                "duration",
                "embedding",
                "sdt",
                "cnumr",
                "nnumr",
                "usr",
                "pclass",
                "edt",
            ]
        ]
        list_df += [df]
    df_jobs = pd.concat(list_df, ignore_index=True)
    df_jobs["pclass"] = df_jobs["pclass"].replace(
        {"compute-bound": 0, "memory-bound": 1}
    )
    df_jobs["tmp"] = ":00"
    df_jobs["sdt_n"] = df_jobs["sdt"].astype(str) + df_jobs["tmp"]
    df_jobs["edt_n"] = df_jobs["edt"].astype(str) + df_jobs["tmp"]

    df_jobs["date_start"] = pd.to_datetime(
        df_jobs["sdt_n"], format="%Y-%m-%d %H:%M:%S%z", errors="coerce"
    )
    df_jobs["date_end"] = pd.to_datetime(
        df_jobs["edt_n"], format="%Y-%m-%d %H:%M:%S%z", errors="coerce"
    )

    df_jobs["duration"] = (df_jobs["duration"] / 60.0).round(0).astype(int)

    df_jobs = df_jobs.drop(columns=["sdt", "tmp", "edt", "sdt_n", "edt_n"])
    return df_jobs


def get_df_for_specific_time_range(
    df_jobs: pd.DataFrame,
    date_time_start_train: datetime,
    date_time_end_train: datetime,
    date_time_start_test: datetime,
    date_time_end_test: datetime,
    max_nb_jobs: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    """Returns a sub dfs of df_jobs, with only jobs run between the provided
    time ranges

    Args:
        df_jobs (pd.DataFrame): The base jobs Dataframe
        date_time_start_train (datetime): Start datetime for train set
        date_time_end_train (datetime): End datetime for train set
        date_time_start_test (datetime): Start datetime for test set
        date_time_end_test (datetime): End datetime for test set
            max_nb_jobs (int | None): If not None, if the number of jobs
            in the train set is > max_nb_jobs then only max_nb_jobs are
            randomly selected

    Returns:
        tuple[pd.DataFrame,pd.DataFrame, np.ndarray, np.ndarray, int]: Returns:
        * The train set Dataframe
        * The test set Dataframe
        * The train set embeddings
        * The test set embeddings
        * The train set size before random selection
    """
    train_df = df_jobs[
        (df_jobs["date_start"] >= date_time_start_train)
        & (df_jobs["date_start"] <= date_time_end_train)
        & (df_jobs["date_end"] >= date_time_start_train)
        & (df_jobs["date_end"] <= date_time_end_train)
    ]

    full_df_size = train_df.shape[0]
    if max_nb_jobs is not None:
        if train_df.shape[0] > max_nb_jobs:
            train_df = train_df.sample(n=max_nb_jobs, random_state=42)

    test_df = df_jobs[
        (df_jobs["date_end"] >= date_time_start_test)
        & (df_jobs["date_end"] <= date_time_end_test)
        & (df_jobs["date_end"] >= date_time_start_test)
        & (df_jobs["date_end"] <= date_time_end_test)
    ]
    embs_train_list = train_df.embedding.tolist()
    embs_train = np.array(embs_train_list)
    embs_test_list = test_df.embedding.tolist()
    embs_test = np.array(embs_test_list)
    train_df = train_df.drop(["embedding"], axis=1)
    test_df = test_df.drop(["embedding"], axis=1)
    if embs_train.shape[0]:
        embs_train = normalize(embs_train, norm="l2")
    else:
        embs_train = None  # type: ignore
    if embs_test.shape[0]:
        embs_test = normalize(embs_test, norm="l2")
    else:
        embs_test = None  # type: ignore
    return train_df, test_df, embs_train, embs_test, full_df_size
