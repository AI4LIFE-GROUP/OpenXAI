import pandas as pd
import numpy as np
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
)
from pandas.core.frame import DataFrame
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
    is_extension_array_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    needs_i8_conversion,
)
import itertools
from pandas.core.reshape import *
from pandas._typing import (
    Dtype
)
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.series import Series


def _get_dummies_1d(
    data,
    prefix,
    prefix_sep="_",
    dummy_na: bool = False,
    sparse: bool = False,
    drop_first: bool = False,
    dtype = None,
) -> DataFrame:
    from pandas.core.reshape.concat import concat

    # Series avoids inconsistent NaN handling
    codes, levels = factorize_from_iterable(Series(data))

    if dtype is None:
        dtype = np.dtype(np.uint8)
    # error: Argument 1 to "dtype" has incompatible type "Union[ExtensionDtype, str,
    # dtype[Any], Type[object]]"; expected "Type[Any]"
    dtype = np.dtype(dtype)  # type: ignore[arg-type]

    if is_object_dtype(dtype):
        raise ValueError("dtype=object is not a valid dtype for get_dummies")

    def get_empty_frame(data) -> DataFrame:
        index: Index | np.ndarray
        if isinstance(data, Series):
            index = data.index
        else:
            index = Index(range(len(data)))
        return DataFrame(index=index)

    # if all NaN
    if not dummy_na and len(levels) == 0:
        return get_empty_frame(data)

    codes = codes.copy()
    if dummy_na:
        codes[codes == -1] = len(levels)
        levels = levels.insert(len(levels), np.nan)

    # if dummy_na, we just fake a nan level. drop_first will drop it again
    if drop_first and len(levels) == 1:
        return get_empty_frame(data)

    number_of_cols = len(levels)

    if prefix is None:
        dummy_cols = levels
    else:
        dummy_cols = Index([f"{prefix}{prefix_sep}{level}" for level in levels])

    index: Index | None
    if isinstance(data, Series):
        index = data.index
    else:
        index = None

    if sparse:

        fill_value: bool | float | int
        if is_integer_dtype(dtype):
            fill_value = 0
        elif dtype == np.dtype(bool):
            fill_value = False
        else:
            fill_value = 0.0

        sparse_series = []
        N = len(data)
        sp_indices: list[list] = [[] for _ in range(len(dummy_cols))]
        mask = codes != -1
        codes = codes[mask]
        n_idx = np.arange(N)[mask]

        for ndx, code in zip(n_idx, codes):
            sp_indices[code].append(ndx)

        if drop_first:
            # remove first categorical level to avoid perfect collinearity
            # GH12042
            sp_indices = sp_indices[1:]
            dummy_cols = dummy_cols[1:]
        for col, ixs in zip(dummy_cols, sp_indices):
            sarr = SparseArray(
                np.ones(len(ixs), dtype=dtype),
                sparse_index=IntIndex(N, ixs),
                fill_value=fill_value,
                dtype=dtype,
            )
            sparse_series.append(Series(data=sarr, index=index, name=col))

        return concat(sparse_series, axis=1, copy=False)

    else:
        # take on axis=1 + transpose to ensure ndarray layout is column-major
        dummy_mat = np.eye(number_of_cols, dtype=dtype).take(codes, axis=1).T

        if not dummy_na:
            # reset NaN GH4446
            dummy_mat[codes == -1] = 0

        if drop_first:
            # remove first GH12042
            dummy_mat = dummy_mat[:, 1:]
            dummy_cols = dummy_cols[1:]
        return DataFrame(dummy_mat, index=index, columns=dummy_cols)
    
    
def convert_categorical_cols(data, feature_types: list, original_columns: list,
                            prefix=None,
                            prefix_sep="_",
                            dummy_na: bool = False,
                            columns=None,
                            sparse: bool = False,
                            drop_first: bool = False,
                            dtype = None
                            ):
    '''
    Converts the categorical columns (specified by feature_types) in df to one-hot
    encoded columns.  
    
    Returns:
        result: pd.DataFrame with one-hot encoded columns
        feature_metadata: dict containing metadata about each feature.
        
        say the original dataframe data has d columns.
            Attributes:
                feature_names: list of str names of length d. used to store the order in which
                    the original features are stored in the returned df result.
                feature_types: list of chars used to store which of the features in the returned
                    df results are continuous vs. discrete.
                    'c' corresponds to continuous features, 'd' corresponds to discrete.
                feature_n_cols: list of ints. feature_n_cols[j] stores the number of columns in one-hot
                    df results that correspond to feature feature_names[j].
                
            
    
    '''
    original_columns = data.columns
    
    ### Copied from the pandas source
    
    from pandas.core.reshape.concat import concat

    dtypes_to_encode = ["object", "category"]

    # if isinstance(data, DataFrame):
    # determine columns being encoded
    if columns is None:
        data_to_encode = data.select_dtypes(include=dtypes_to_encode)
    elif not is_list_like(columns):
        raise TypeError("Input must be a list-like for parameter `columns`")
    else:
        data_to_encode = data[columns]

    # validate prefixes and separator to avoid silently dropping cols
    def check_len(item, name):

        if is_list_like(item):
            if not len(item) == data_to_encode.shape[1]:
                len_msg = (
                    f"Length of '{name}' ({len(item)}) did not match the "
                    "length of the columns being encoded "
                    f"({data_to_encode.shape[1]})."
                )
                raise ValueError(len_msg)

    check_len(prefix, "prefix")
    check_len(prefix_sep, "prefix_sep")

    if isinstance(prefix, str):
        prefix = itertools.cycle([prefix])
    if isinstance(prefix, dict):
        prefix = [prefix[col] for col in data_to_encode.columns]

    if prefix is None:
        prefix = data_to_encode.columns

    # validate separators
    if isinstance(prefix_sep, str):
        prefix_sep = itertools.cycle([prefix_sep])
    elif isinstance(prefix_sep, dict):
        prefix_sep = [prefix_sep[col] for col in data_to_encode.columns]

    with_dummies: list[DataFrame]
    if data_to_encode.shape == data.shape:
        # Encoding the entire df, do not prepend any dropped columns
        with_dummies = []
    elif columns is not None:
        # Encoding only cols specified in columns. Get all cols not in
        # columns to prepend to result.
        with_dummies = [data.drop(columns, axis=1)]
    else:
        # Encoding only object and category dtype columns. Get remaining
        # columns to prepend to result.
        with_dummies = [data.select_dtypes(exclude=dtypes_to_encode)]


    num_dummies_per_col = {}
    
    ### CHANGED FROM SOURCE CODE
    
    new_feature_names = list(with_dummies[0].columns)
    # all features thus far are continuous
    new_feature_types = ['c' for i in range(len(new_feature_names))]
    # all features thus far have 1 column
    new_feature_n_cols = [1 for i in range(len(new_feature_names))]

    for (col, pre, sep) in zip(data_to_encode.items(), prefix, prefix_sep):
        # col is (column_name, column), use just column data here
        dummy = _get_dummies_1d(
            col[1],
            prefix=pre,
            prefix_sep=sep,
            dummy_na=dummy_na,
            sparse=sparse,
            drop_first=drop_first,
            dtype=dtype,
        )
        
        # append column name
        new_feature_names.append(pre)
        # new discrete feature
        new_feature_types.append('d')
        # append number of dummy columns created
        new_feature_n_cols.append(len(dummy.columns))


        with_dummies.append(dummy)
    result = concat(with_dummies, axis=1)
    
    assert sum(new_feature_n_cols) == len(result.columns)
    
    feature_metadata = {}
    feature_metadata['feature_names'] = new_feature_names
    feature_metadata['feature_types'] = new_feature_types
    feature_metadata['feature_n_cols'] = new_feature_n_cols
    
    return result, feature_metadata