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
        list_rpr_labels (np.ndarray): The labels associated to each
        representative vector
        list_rpr_embeddings (np.ndarray): The representative embedding vector
        emb (np.ndarray): The embedding for which we are looking for close
        cluster
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
    label_to_df_values: dict[int, Any],
    strategy: str,
    list_rpr_labels: np.ndarray,
    list_rpr_embeddings: np.ndarray,
    row: pd.Series,
    emb: np.ndarray,
    eps: float,
) -> tuple[Any, Any, Any, Any] | None:
    """Makes prediction for a given job (row, emb)

    Args:
        label_to_df_values (dict[int, Any]): The label dictionary according
            the strategy
        strategy (str): The selected strategy
        list_rpr_labels (np.ndarray): The labels associated to each
        representative
        vector
        list_rpr_embeddings (np.ndarray): The representative embedding vector
        row (pd.Series): A jobs DataFrame row
        emb (np.ndarray): The embedding corresponding the the job row
        eps (float): The eps value to consider the job close enough of a
        cluster

    Returns:
        tuple[tuple[Any, Any, Any, Any] | None:
        (predicted runtime, target runtime, predicted pclass, target pclass)
    """
    if strategy not in ["as", "su", "sr", "sur"]:
        raise KeyError(
            "Unknown Strategy: availables ones are 'as', 'sr', 'su', and 'sur'"
        )
    label = get_close_label(list_rpr_labels, list_rpr_embeddings, emb, eps)
    if label is not None:
        if strategy == "as":
            return (
                label_to_df_values[label][0],
                row[1]["duration"],
                label_to_df_values[label][1],
                row[1]["pclass"],
            )
        test_cnumr = row[1]["cnumr"]
        test_nnumr = row[1]["nnumr"]
        test_usr = row[1]["usr"]
        if strategy == "sr":
            if (
                test_nnumr in label_to_df_values[label].keys()
                and test_cnumr in label_to_df_values[label][test_nnumr].keys()
            ):
                return (
                    label_to_df_values[label][test_nnumr][test_cnumr][0],
                    row[1]["duration"],
                    label_to_df_values[label][test_nnumr][test_cnumr][1],
                    row[1]["pclass"],
                )
        if strategy == "su":
            if test_usr in label_to_df_values[label].keys():
                return (
                    label_to_df_values[label][test_usr][0],
                    row[1]["duration"],
                    label_to_df_values[label][test_usr][1],
                    row[1]["pclass"],
                )
        if strategy == "sur":
            if (
                test_nnumr in label_to_df_values[label].keys()
                and test_cnumr in label_to_df_values[label][test_nnumr].keys()
                and test_usr
                in label_to_df_values[label][test_nnumr][test_cnumr].keys()
            ):
                return (
                    label_to_df_values[label][test_nnumr][test_cnumr][
                        test_usr
                    ][0],
                    row[1]["duration"],
                    label_to_df_values[label][test_nnumr][test_cnumr][
                        test_usr
                    ][1],
                    row[1]["pclass"],
                )
    return None


def manage_predict(
    label_to_df_values: dict[int, Any],
    strategy: str,
    list_rpr_labels: np.ndarray,
    list_rpr_embeddings: np.ndarray,
    row: pd.Series,
    emb: np.ndarray,
    eps_euclidean: float,
) -> (
    tuple[Any | None, Any | None, Any | None, Any | None, Any, Any, Any] | None
):
    """Launches a prediction for a given job (row, emb), collects the results
    and returns them

    Args:
        label_to_df_values (dict[int, Any]): The label dictionary according
            the strategy
        strategy (str): The selected strategy
        list_rpr_labels (np.ndarray): The labels associated to each
        representative vector
        list_rpr_embeddings (np.ndarray): The representative embedding vector
        row (pd.Series): A jobs DataFrame row
        emb (np.ndarray): The embedding corresponding the the job row
        eps_euclidean (float): The eps value to consider the job close enough
        of a cluster

    Returns:
        tuple[Any | None, Any | None, Any | None, Any | None, Any, Any,
        Any] | None:
        (predicted runtime, target runtime, predicted pclass, target pclass,
        job id, job start datetime, job end datetime)

    """
    out = predicts(
        label_to_df_values,
        strategy,
        list_rpr_labels,
        list_rpr_embeddings,
        row,
        emb,
        eps=eps_euclidean,
    )

    if out is None:
        pred_time, gt_time, pred_pclass, gt_pclass = None, None, None, None
    else:
        pred_time, gt_time, pred_pclass, gt_pclass = out

    output = None
    if gt_time is not None:
        output = (
            gt_time,
            pred_time,
            gt_pclass,
            pred_pclass,
            row[1]["jid"],
            row[1]["date_start"],
            row[1]["date_end"],
        )
    return output
