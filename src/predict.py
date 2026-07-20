import math
from typing import Any

import numpy as np
import pandas as pd  # type: ignore


def get_close_label(
    list_rpr_labels: np.ndarray,
    list_rpr_embeddings: np.ndarray,
    emb: np.ndarray,
    eps: float,
) -> int | None:
    """Returns the label of the closest cluster if close enough (<= eps)

    Args:
        list_rpr_labels (np.ndarray): The labels associated to each representative
        vector
        list_rpr_embeddings (np.ndarray): The representative embedding vector
        emb (np.ndarray): The embedding for which we are looking for close cluster
        eps (float): _description_

    Returns:
        int | None: The label of the closest cluster if found else None
    """
    distances = np.linalg.norm(emb - list_rpr_embeddings, axis=1)
    argmin_index = np.argmin(distances)
    argmin_label = list_rpr_labels[argmin_index]
    distance = distances[argmin_index]
    distance = 0.0 if math.isclose(distance, 0.0) else distance
    if distance > eps or argmin_label == -1:
        return None
    return argmin_label


def predicts(
    label_to_df_values_sr: dict[int, Any],
    label_to_df_values_as: dict[int, Any],
    list_rpr_labels: np.ndarray,
    list_rpr_embeddings: np.ndarray,
    row: pd.Series,
    emb: np.ndarray,
    eps: float,
) -> tuple[tuple[Any, Any, Any, Any] | None, tuple[Any, Any, Any, Any] | None]:
    """Makes prediction for a given job (row, emb)

    Args:
        label_to_df_values_sr (dict[int, Any]): The same resources jobs dictionary
        label_to_df_values_as (dict[int, Any]): The all similar jobs dictionary
        list_rpr_labels (np.ndarray): The labels associated to each representative
        vector
        list_rpr_embeddings (np.ndarray): The representative embedding vector
        row (pd.Series): A jobs DataFrame row
        emb (np.ndarray): The embedding corresponding the the job row
        eps (float): The eps value to consider the job close enough of a cluster

    Returns:
        tuple[tuple[Any, Any, Any, Any] | None, tuple[Any, Any, Any, Any] | None]:
        (
        (predicted runtime, target runtime, predicted pclass, target pclass)
            // for same resources jobs
        (predicted runtime, target runtime, predicted pclass, target pclass)
            // for all similar jobs
        )
    """
    output_as = None
    output_sr = None
    label = get_close_label(list_rpr_labels, list_rpr_embeddings, emb, eps)
    if label is not None:
        output_as = (
            label_to_df_values_as[label][0],
            row[1]["duration"],
            label_to_df_values_as[label][1],
            row[1]["pclass"],
        )
        test_cnumr = row[1]["cnumr"]
        test_nnumr = row[1]["nnumr"]
        if (
            test_nnumr in label_to_df_values_sr[label].keys()
            and test_cnumr in label_to_df_values_sr[label][test_nnumr].keys()
        ):
            output_sr = (
                label_to_df_values_sr[label][test_nnumr][test_cnumr][0],
                row[1]["duration"],
                label_to_df_values_sr[label][test_nnumr][test_cnumr][1],
                row[1]["pclass"],
            )
    return output_sr, output_as


def manage_predict(
    label_to_df_values_sr: dict[int, Any],
    label_to_df_values_as: dict[int, Any],
    list_rpr_labels: np.ndarray,
    list_rpr_embeddings: np.ndarray,
    row: pd.Series,
    emb: np.ndarray,
    eps_euclidean: float,
) -> tuple[
    tuple[Any | None, Any | None, Any | None, Any | None, Any, Any, Any] | None,
    tuple[Any | None, Any | None, Any | None, Any | None, Any, Any, Any] | None,
]:
    """Launches a prediction for a given job (row, emb), collects the results and
    returns them

    Args:
        label_to_df_values_sr (dict[int, Any]): The same resources jobs dictionary
        label_to_df_values_as (dict[int, Any]): The all similar jobs dictionary
        list_rpr_labels (np.ndarray): The labels associated to each representative
        vector
        list_rpr_embeddings (np.ndarray): The representative embedding vector
        row (pd.Series): A jobs DataFrame row
        emb (np.ndarray): The embedding corresponding the the job row
        eps_euclidean (float): The eps value to consider the job close enough of a
            cluster

    Returns:
        tuple[ tuple[Any | None, Any | None, Any | None, Any | None, Any, Any, Any]
            | None,
        tuple[Any | None, Any | None, Any | None, Any | None, Any, Any, Any] | None, ]:
        (
        (predicted runtime, target runtime, predicted pclass, target pclass, job id,
            job start datetime, job end datetime)
            // for same resources jobs
        (predicted runtime, target runtime, predicted pclass, target pclass, job id,
            job start datetime, job end datetime)
            // for all similar jobs
        )
    """
    out_sr, out_as = predicts(
        label_to_df_values_sr,
        label_to_df_values_as,
        list_rpr_labels,
        list_rpr_embeddings,
        row,
        emb,
        eps=eps_euclidean,
    )
    if out_sr is None:
        pred_time, gt_time, pred_pclass, gt_pclass = None, None, None, None
    else:
        pred_time, gt_time, pred_pclass, gt_pclass = out_sr
    if out_as is None:
        (
            pred_time_all_sim,
            gt_time_all_sim,
            pred_pclass_all_sim,
            gt_pclass_all_sim,
        ) = (None, None, None, None)
    else:
        (
            pred_time_all_sim,
            gt_time_all_sim,
            pred_pclass_all_sim,
            gt_pclass_all_sim,
        ) = out_as

    output_sr = None
    output_as = None
    if gt_time is not None:
        output_sr = (
            gt_time,
            pred_time,
            gt_pclass,
            pred_pclass,
            row[1]["jid"],
            row[1]["date_start"],
            row[1]["date_end"],
        )
    if gt_time_all_sim is not None:
        output_as = (
            gt_time_all_sim,
            pred_time_all_sim,
            gt_pclass_all_sim,
            pred_pclass_all_sim,
            row[1]["jid"],
            row[1]["date_start"],
            row[1]["date_end"],
        )
    return output_sr, output_as
