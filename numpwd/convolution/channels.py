"""Utility tools for convoluting channels."""
from typing import List, Tuple, Optional
from pandas import DataFrame, merge
from numpy import ndarray


def get_channel_overlap_indices(
    channels1: DataFrame,
    channels2: DataFrame,
    columns: Optional[List[str]] = [
        "l_o",
        "s_o",
        "j_o",
        "mj_o",
        "l_i",
        "s_i",
        "j_i",
        "mj_i",
    ],
) -> Tuple[ndarray, ndarray]:
    """Computes overlapping channel indicies."""
    for df in [channels1, channels2]:
        missing_columns = set(columns) - set(df.columns)
        if missing_columns:
            raise KeyError(
                "Channels DataFrame does not contain all merge columns."
                "\nMissing columns `%s`" % missing_columns
            )
    id1_name = channels1.index.name
    id2_name = channels2.index.name
    return (
        merge(
            channels1.reset_index().rename(columns={id1_name: "id"}),
            channels2.reset_index().rename(columns={id2_name: "id"}),
            how="inner",
            left_on=columns,
            right_on=columns,
            suffixes=["1", "2"],
        )
        .rename(columns={"id": "id_dens", "index": "id_op"})[["id1", "id2"]]
        .values.T
    )
