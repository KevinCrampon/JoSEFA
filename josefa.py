"""The JoSEFA Script"""

import argparse
import os
from datetime import datetime, timedelta
from time import time
from typing import Any

import numpy as np
import pandas as pd  # type: ignore
from hdbscan import HDBSCAN  # type: ignore
from sklearn.decomposition import PCA  # type: ignore

from src.clustering import get_dicts_pred, get_representative  # type: ignore
from src.dataframe_tools import (  # type: ignore
    get_df_for_specific_time_range,
    load_df,
)
from src.files_management import (  # type: ignore
    create_dir,
    write_clustering_results,
    write_global_result_header,
    write_global_result_line,
)
from src.metrics import (  # type: ignore
    compute_metrics_classification,
    compute_metrics_clustering,
    compute_metrics_regression,
)
from src.predict import manage_predict  # type: ignore

pd.set_option("future.no_silent_downcasting", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "-parquet_dir", type=str, help="Path to the parquet files directory"
    )
    parser.add_argument("-year_month", type=str, help="format: YYYY-MM")
    parser.add_argument(
        "-nb_history_days",
        "-n",
        type=int,
        help="The number of history days used to train the model",
    )
    parser.add_argument(
        "-retrain_each_days",
        "-r",
        type=int,
        help="Retrain each nb days, is also the number of days used to test "
        "the model",
    )
    parser.add_argument(
        "-strategy",
        "-s",
        type=str,
        choices=["as", "sr", "su", "sur"],
        help="The strategy used to perform prediction",
    )

    args = parser.parse_args()
    year_month = args.year_month
    parquet_dir = args.parquet_dir
    nb_history_day = args.nb_history_days
    retrain_each_days = args.retrain_each_days
    strategy = args.strategy
    dir_path = create_dir(
        f"res_{strategy}_hdbscan_all_data_",
        retrain_each_days,
        nb_history_day,
    )
    file_path = f"{dir_path}/{year_month}.csv"
    year_month_dir = f"{dir_path}/{year_month}"
    if not os.path.isdir(year_month_dir):
        os.mkdir(year_month_dir)

    df_dir = f"{year_month_dir}/dfs"
    if not os.path.isdir(df_dir):
        os.mkdir(df_dir)
    write_global_result_header(file_path=file_path)

    offset = "+09:00"
    date_format = "%Y-%m-%d %H:%M:%S%z"
    date_start = f"{year_month}-01 00:00:00{offset}"
    year = year_month.split("-")[0]
    month = year_month.split("-")[-1]
    train_start_datetime = datetime.strptime(date_start, date_format)
    if month in ["01", "03", "05", "07", "08", "10", "12"]:
        last_day = "31"
    elif month in ["04", "06", "09", "11"]:
        last_day = "30"
    else:
        if year == "2024":
            last_day = "29"
        else:
            last_day = "28"

    until_date = datetime.strptime(
        f"{year_month}-{last_day} 23:59:59{offset}", date_format
    )

    delta = until_date - train_start_datetime
    nb_splits = int((delta.days + 1) / retrain_each_days)
    print("Nb splits: ", nb_splits)
    i_split = 0
    df_global_jobs = load_df(
        parquet_dir, year_month.replace("-", "_").replace("20", "")
    )
    while train_start_datetime < until_date:
        i_split += 1
        train_ok = False
        test_ok = False
        ground_truth_time: list[float] = []
        predicted_time: list[float] = []
        ground_truth_pclass: list[float] = []
        predicted_pclass: list[float] = []
        list_jid: list[str] = []
        list_date_start: list[str] = []
        list_date_end: list[str] = []
        train_end_datetime = train_start_datetime + timedelta(
            days=nb_history_day, seconds=-1
        )
        test_start_datetime = train_end_datetime + timedelta(seconds=1)
        test_end_datetime = test_start_datetime + timedelta(
            days=retrain_each_days, seconds=-1
        )

        train_date_start = train_start_datetime.strftime(date_format)
        train_date_end = train_end_datetime.strftime(date_format)
        test_date_start = test_start_datetime.strftime(date_format)
        test_date_end = test_end_datetime.strftime(date_format)

        print(
            f"{datetime.now().strftime(date_format)} : "
            f"Split: {i_split} / {nb_splits}"
            "\n\tTrain:"
            f"\n\t\tFrom : {train_start_datetime.strftime(date_format)}"
            f"\n\t\tTo : {train_end_datetime.strftime(date_format)}"
            "\n\tTest:"
            f"\n\t\tFrom : {test_start_datetime.strftime(date_format)}"
            f"\n\t\tTo : {test_end_datetime.strftime(date_format)}"
        )

        max_nb_jobs = 500000

        train_df, test_df, embs_train, embs_test, nb_base_train_jobs = (
            get_df_for_specific_time_range(
                df_global_jobs,
                train_start_datetime,
                train_end_datetime,
                test_start_datetime,
                test_end_datetime,
                max_nb_jobs,
            )
        )

        min_cluster_size = 200
        if train_df.shape[0] > 0:
            train_ok = True

            print(
                f"{datetime.now().strftime(date_format)} : "
                f"Train size: {train_df.shape[0]}"
            )
            print(
                f"{datetime.now().strftime(date_format)} : "
                f"Test size: {test_df.shape[0]}"
            )
            print(
                f"{datetime.now().strftime(date_format)} : "
                f"Min Cluster Size: {min_cluster_size}"
            )

            eps_cosine = 0.1
            eps_euclidean = np.sqrt(2 * (1 - eps_cosine))

            clustering = HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric="euclidean",
            )
            pca = PCA(
                n_components=50,
                random_state=42,
            )

            time_start_pca = time()
            X_reduced = pca.fit_transform(embs_train)
            time_end_pca = time()
            time_start_clustering = time()
            clustering.fit(X_reduced)
            time_end_clustering = time()
            train_df["label"] = clustering.labels_

            dict_clustering_metrics = compute_metrics_clustering(
                embs_train, clustering.labels_, clustering
            )
            dict_clustering_metrics_reduced = compute_metrics_clustering(
                X_reduced, clustering.labels_, clustering
            )

            if test_df.shape[0] > 0:
                test_ok = True

                list_rpr_labels, list_rpr_embeddings = get_representative(
                    train_df, embs_train
                )

                dict_predictions = get_dicts_pred(train_df, strategy)

                print(
                    f"{datetime.now().strftime(date_format)}",
                    " : Launch predictions...",
                )

                nb_not_predicted: int | None = 0
                # for MyPy checking
                assert nb_not_predicted is not None
                list_test_embs = embs_test.tolist()

                list_params: list[Any] = list()
                results = []
                for row, emb in zip(test_df.iterrows(), list_test_embs):
                    results.append(
                        manage_predict(
                            dict_predictions,
                            strategy,
                            list_rpr_labels,
                            list_rpr_embeddings,
                            row,
                            emb,
                            eps_euclidean,
                        )
                    )

                print(
                    f"{datetime.now().strftime(date_format)} : "
                    "Analyse predictions..."
                )
                for res in results:
                    # res_sr, res_as, res_su, res_sur = res
                    if (
                        res is None
                        or res[0] is None
                        or res[1] is None
                        or res[2] is None
                        or res[3] is None
                    ):
                        nb_not_predicted += 1
                    else:
                        ground_truth_time += [res[0]]
                        predicted_time += [res[1]]
                        ground_truth_pclass += [res[2]]
                        predicted_pclass += [res[3]]
                        list_jid += [res[4]]
                        list_date_start += [res[5]]
                        list_date_end += [res[6]]

                print(
                    f"{datetime.now().strftime(date_format)} : "
                    "Write predictions..."
                )
                # Bound
                write_clustering_results(
                    file_path=f"{df_dir}/"
                    f"df_pclass_year-month-{year_month}_split-"
                    f"{i_split}.csv",
                    pred=predicted_pclass,
                    gt=ground_truth_pclass,
                    list_jid=list_jid,
                    list_date_start=list_date_start,
                    list_date_end=list_date_end,
                )

                # Time
                write_clustering_results(
                    file_path=f"{df_dir}/df_time_year-month-{year_month}_"
                    f"split-{i_split}.csv",
                    pred=predicted_time,
                    gt=ground_truth_time,
                    list_jid=list_jid,
                    list_date_start=list_date_start,
                    list_date_end=list_date_end,
                )

                r2_time, mae_time, mse_time = compute_metrics_regression(
                    predicted_time, ground_truth_time
                )

                acc_pclass, f1_pclass = compute_metrics_classification(
                    predicted_pclass, ground_truth_pclass
                )

        if not train_ok:
            time_start_pca = 0
            time_end_pca = 0
            time_start_clustering = 0
            time_end_clustering = 0
            nb_base_train_jobs = 0
        if not train_ok or not test_ok:
            r2_time = None
            mae_time = None
            mse_time = None
            acc_pclass = None
            f1_pclass = None
            nb_not_predicted = None

        write_global_result_line(
            file_path=file_path,
            train_start_datetime=train_start_datetime,
            train_end_datetime=train_end_datetime,
            test_start_datetime=test_start_datetime,
            test_end_datetime=test_end_datetime,
            nb_base_train_jobs=nb_base_train_jobs,
            nb_actual_train_jobs=train_df.shape[0],
            nb_test_jobs=test_df.shape[0],
            r2_time=r2_time,
            mae_time=mae_time,
            mse_time=mse_time,
            acc_pclass=acc_pclass,
            f1_pclass=f1_pclass,
            nb_not_predicted=nb_not_predicted,
            n_clusters=dict_clustering_metrics["n_clusters"],
            noise_ratio=dict_clustering_metrics["noise_ratio"],
            silhouette=dict_clustering_metrics["silhouette"],
            calinski_harabasz=dict_clustering_metrics["calinski_harabasz"],
            davies_bouldin=dict_clustering_metrics["davies_bouldin"],
            dbcv=None,  # Extracted from model trained on reduced data
            # Extracted from model trained on reduced data
            cluster_persistence=None,
            reduced_noise_ratio=dict_clustering_metrics_reduced["noise_ratio"],
            reduced_silhouette=dict_clustering_metrics_reduced["silhouette"],
            reduced_calinski_harabasz=dict_clustering_metrics_reduced[
                "calinski_harabasz"
            ],
            reduced_davies_bouldin=dict_clustering_metrics_reduced[
                "davies_bouldin"
            ],
            reduced_dbcv=dict_clustering_metrics_reduced["dbcv"],
            reduced_cluster_persistence=dict_clustering_metrics_reduced[
                "cluster_persistence"
            ],
            pca_time=f"{time_end_pca-time_start_pca:.2f}",
            clustering_time=f"{time_end_clustering-time_start_clustering:.2f}",
            min_cluster_size=min_cluster_size,
            date_format=date_format,
        )

        train_start_datetime = train_start_datetime + timedelta(
            days=retrain_each_days
        )

    print("Finished")
