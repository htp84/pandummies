import pandas as pd
from pandummies import __version__, create_dummies, from_dummies

prefix_basic = ["col1", "col2"]
df_basic = pd.DataFrame({"A": ["a", "b", "a"], "B": ["b", "a", "c"], "C": [1, 2, 3]})
dummies_basic = pd.get_dummies(df_basic, prefix=prefix_basic)
columns_basic = df_basic.columns.tolist()


def _iris():
    iris = (
        pd.read_csv("data/iris.csv")
        .drop(["embarked", "who", "alive"], axis=1)
        .dropna(axis=0)
    )
    columns = iris.columns.tolist()
    dtypes = iris.dtypes.tolist()
    index = iris.index.tolist()
    return iris, columns, dtypes, index


def _iris_dummies(prefix_sep):
    iris = _iris()[0]
    iris_dummies = pd.get_dummies(iris, prefix_sep=prefix_sep)
    return iris_dummies


def test_version():
    assert __version__ == "0.1.0"


def test_type_basic():
    dummies = pd.get_dummies(df_basic, prefix=prefix_basic)
    assert type(from_dummies(dummies_basic)) == pd.core.frame.DataFrame


def test_not_equal_basic():
    dummies = pd.get_dummies(df_basic, prefix=prefix_basic)
    assert dummies_basic.equals(df_basic) == False


def test_equal_basic():
    dummies = pd.get_dummies(df_basic, prefix=prefix_basic)
    new_df = from_dummies(df_basic, column_order=columns_basic)
    assert new_df.equals(new_df) == True


def test_not_equal_iris():
    iris, _, __, ___ = _iris()
    print(iris.head())
    iris_dummies = _iris_dummies(prefix_sep="#")
    new_iris = from_dummies(iris_dummies, prefix_sep="#")
    assert new_iris.equals(iris) == False


def test_not_equal_column_iris():
    iris, columns, _, __ = _iris()
    iris_dummies = _iris_dummies(prefix_sep="#")
    new_iris = from_dummies(iris_dummies, prefix_sep="#", column_order=columns)
    assert new_iris.equals(iris) == False


def test_not_equal_column_dtype_iris():
    iris, columns, dtype, _ = _iris()
    iris_dummies = _iris_dummies(prefix_sep="#")
    new_iris = from_dummies(
        iris_dummies, prefix_sep="#", column_order=columns, dtypes=dtype
    )
    assert new_iris.equals(iris) == False


def test_equal_column_dtype_index_iris():
    iris, columns, dtype, index = _iris()
    iris_dummies = _iris_dummies(prefix_sep="#")
    new_iris = from_dummies(
        iris_dummies, prefix_sep="#", column_order=columns, dtypes=dtype, index=index
    )
    assert new_iris.equals(iris) == True
