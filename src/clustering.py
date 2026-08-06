from typing import Any

import numpy as np
import pandas as pd  # type: ignore


def get_dicts_pred_as(label_to_df: dict[int, Any]) -> dict[int, Any]:
    """Creates from Dataframe with clustering label:
    * A dictionary where keys are label
    and value are
       * Runtime median
       * Class majority
    * A dictionary where keys are label and value are
       * Runtime median
       * Class majority
    For the All Similar strategy

    Args:
        label_to_df (dict[int, Any]): The dict from dataframe label split

    Returns:
        dict[int, Any]: The dictionary
    """
    label_to_df_values_as: dict[int, Any] = {}
    for k in label_to_df.keys():
        label_to_df_values_as[k] = (
            label_to_df[k]["duration"].median(),
            int(round(label_to_df[k]["pclass"].mean(), 0)),
        )
    return label_to_df_values_as


def get_dict_preds_sr(label_to_df: dict[int, Any]) -> dict[int, Any]:
    """Creates from Dataframe with clustering label:
    * A dictionary where keys are label, and subkeys are resources requested
    and value are
       * Runtime median
       * Class majority
    * A dictionary where keys are label and value are
       * Runtime median
       * Class majority
    For the Same Resources strategy

    Args:
        label_to_df (dict[int, Any]): The dict from dataframe label split

    Returns:
        dict[int, Any]: The dictionary
    """
    label_to_df_values_sr: dict[int, Any] = {}
    for k in label_to_df.keys():
        label_to_df_values_sr[k] = {}
        groups_nnumr = label_to_df[k].groupby(["nnumr"])
        for nnumr, group_nnumr in groups_nnumr:
            nnumr = nnumr[0]
            label_to_df_values_sr[k][nnumr] = {}
            groups_cnumr = group_nnumr.groupby(["cnumr"])
            for cnumr, group_cnumr in groups_cnumr:
                cnumr = cnumr[0]
                label_to_df_values_sr[k][nnumr][cnumr] = (
                    group_cnumr["duration"].median(),
                    int(round(group_cnumr["pclass"].mean(), 0)),
                )
    return label_to_df_values_sr


def get_dict_preds_su(label_to_df: dict[int, Any]) -> dict[int, Any]:
    """Creates from Dataframe with clustering label:
    * A dictionary where keys are label, and subkeys are user
    and value are
       * Runtime median
       * Class majority
    * A dictionary where keys are label and value are
       * Runtime median
       * Class majority
    For the Same User strategy

    Args:
        label_to_df (dict[int, Any]): The dict from dataframe label split

    Returns:
        dict[int, Any]: The dictionary
    """
    label_to_df_values_su: dict[int, Any] = {}
    for k in label_to_df.keys():
        label_to_df_values_su[k] = {}
        groups_usr = label_to_df[k].groupby(["usr"])
        for usr, group_usr in groups_usr:
            usr = usr[0]
            label_to_df_values_su[k][usr] = (
                group_usr["duration"].median(),
                int(round(group_usr["pclass"].mean(), 0)),
            )
    return label_to_df_values_su


def get_dict_preds_sur(label_to_df: dict[int, Any]) -> dict[int, Any]:
    """Creates from Dataframe with clustering label:
    * A dictionary where keys are label, and subkeys are resources and user
    and value are
       * Runtime median
       * Class majority
    * A dictionary where keys are label and value are
       * Runtime median
       * Class majority
    For the Same User AND Resources strategy

    Args:
        label_to_df (dict[int, Any]): The dict from dataframe label split

    Returns:
        dict[int, Any]: The dictionary
    """
    label_to_df_values_sur: dict[int, Any] = {}
    for k in label_to_df.keys():
        label_to_df_values_sur[k] = {}
        groups_nnumr = label_to_df[k].groupby(["nnumr"])
        for nnumr, group_nnumr in groups_nnumr:
            nnumr = nnumr[0]
            label_to_df_values_sur[k][nnumr] = {}
            groups_cnumr = group_nnumr.groupby(["cnumr"])
            for cnumr, group_cnumr in groups_cnumr:
                cnumr = cnumr[0]
                label_to_df_values_sur[k][nnumr][cnumr] = {}
                groups_usr = group_cnumr.groupby(["usr"])
                for usr, group_usr in groups_usr:
                    usr = usr[0]
                    label_to_df_values_sur[k][nnumr][cnumr][usr] = (
                        group_usr["duration"].median(),
                        int(round(group_usr["pclass"].mean(), 0)),
                    )
    return label_to_df_values_sur


def get_dicts_pred(
    df: pd.DataFrame,
    strategy: str,
) -> dict[int, Any]:
    """Creates from Dataframe with clustering label:
    * A dictionary where keys are label, and subkeys are resources requested
    and value are
       * Runtime median
       * Class majority
    * A dictionary where keys are label and value are
       * Runtime median
       * Class majority

    Args:
        df (pd.DataFrame): The Dataframe of jobs with clustering labels added
        strategy (str): The strategy used to perform prediction

    Returns:
        dict[int, Any]: The dictionary
    """
    label_to_df = {label: group_df for label, group_df in df.groupby("label")}
    if strategy == "as":
        return get_dicts_pred_as(label_to_df)
    if strategy == "sr":
        return get_dict_preds_sr(label_to_df)
    if strategy == "su":
        return get_dict_preds_su(label_to_df)
    if strategy == "sur":
        return get_dict_preds_sur(label_to_df)
    raise KeyError(
        "Unknown Strategy: available ones are 'as', 'sr', 'su', and 'sur'"
    )


def get_representative(
    df_train: pd.DataFrame, embs_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Computes a representative vector for each cluster

    Args:
        df_train (pd.DataFrame): The jobs dataframe with clustering label
        embs_train (np.ndarray): The embedding of each job

    Returns:
        tuple[np.ndarray, np.ndarray]:
            * The label id list
            * The representative vector corresponding
    """
    df_tmp = pd.DataFrame(
        {"label": df_train["label"], "embedding": embs_train.tolist()}
    )
    list_rpr_labels = []
    list_rpr_embeddings = []
    for i, group in df_tmp.groupby("label"):
        label_i = i
        list_rpr_labels += [label_i]
        embs = np.vstack(group["embedding"])
        emb_mean = embs.mean(axis=0)
        list_rpr_embeddings += [emb_mean]
    return np.array(list_rpr_labels), np.vstack(list_rpr_embeddings)
