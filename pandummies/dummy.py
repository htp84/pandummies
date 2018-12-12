try:
    import pandas as pd
except ImportError as exc:
    print(exc)
    print(f"The module {exc.name} is required!")


def test_na(df: pd.DataFrame) -> None:
    if df.isnull().values.any():
        raise ValueError(
            "The dataframe contains missing data (NaN). This is not aloud."
        )


def _create_dict(columns, values):
    d = {}
    for i in columns:
        for j in values:
            if i != j[0]:
                continue
            if i == j[0] and i not in d.keys():
                d[i] = [j[1]]
            else:
                d[i].append(j[1])
    return d


def from_dummies(dummy_df, prefix_sep="_", column_order=None, dtypes=None, index=None):
    """
    as for the get_dummies functions only categorical columns should be used. Is for
    example a boolean column used the result will be bad.

    The purpose of this function is not to recreate the original dataframe. It is to create
    at dataframe that contains, for example, prediction. Then the this transformed datfarme is more
    readable.
    """
    test_na(dummy_df)
    stacked = pd.DataFrame(dummy_df.stack()).reset_index()
    category = stacked[
        (stacked["level_1"].str.contains(prefix_sep)) & (stacked[0] == 1)
    ]
    regular = stacked[~stacked["level_1"].str.contains(prefix_sep)]
    category_column_value = category.level_1.str.split(prefix_sep).tolist()
    category_columns = list(set([i[0] for i in category_column_value]))
    regular_column_value = regular[["level_1", 0]].values.tolist()
    regular_columns = regular["level_1"].unique().tolist()
    a = _create_dict(category_columns, category_column_value)
    b = _create_dict(regular_columns, regular_column_value)
    c = dict(a, **b)
    df = pd.DataFrame(c)
    if index:
        df.index = index
    if column_order:
        df = df[column_order]
    if dtypes:
        for i, j in zip(df.columns, dtypes):
            df[i] = df[i].astype(j)
    return df


def create_dummies(
    source: pd.DataFrame,
    dummies: list,
    iloc: bool = False,
    target: pd.DataFrame = pd.DataFrame(),
    prefix: bool = False,
) -> pd.DataFrame:
    """
    Creates dummies based on columns given in parameter dummies. If iloc is
    True index can be given
    """
    if iloc:
        columns = source.columns
        dummies = [columns[i] for i in dummies]
    if not target.empty:
        df = target
    else:
        df = pd.DataFrame()
    for i in dummies:
        if not prefix:
            prefix, prefix_sep = "", ""
        else:
            prefix, prefix_sep = i, "_"
        _temp = pd.get_dummies(
            data=source.loc[:, i], prefix=prefix, prefix_sep=prefix_sep
        )
        df = pd.concat([df, _temp], axis=1)
    return df
