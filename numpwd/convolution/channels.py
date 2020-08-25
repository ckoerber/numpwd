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
    """Computes overlapping channel indicies.

    Runs inner merge (join) on specfied columns and returns indices of
    overlapping channels.
    """
    for key, df in {"First": channels1, "Second": channels2}.items():
        missing_columns = set(columns) - set(df.columns)
        if missing_columns:
            raise KeyError(
                f"{key} channels DataFrame does not contain all merge columns."
                f" Missing columns {missing_columns}"
            )
    id1_name = channels1.index.name or "index"
    id2_name = channels2.index.name or "index"
    return tuple(
        merge(
            channels1.reset_index().rename(columns={id1_name: "id1"}),
            channels2.reset_index().rename(columns={id2_name: "id2"}),
            how="inner",
            left_on=columns,
            right_on=columns,
            suffixes=["1", "2"],
        )[["id1", "id2"]].values.T
    )
