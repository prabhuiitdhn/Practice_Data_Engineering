# https://gitlab.com/marvelousmlops/cicd-for-mlops/-/tree/part1_setting-up-your-first-ci-pipeline?ref_type=heads


from pandas import DataFrame


def filter_data(data: DataFrame, column: str, threshold: int) -> DataFrame:
    """
    Filter data based on threshold of column value.
    :param data: A dataframe containing the data
    :param column: The name of the column that will be used for filtering
    :param threshold: The threshold value that will be used for filtering
    :return: A dataframe containing the filtered data
    """
    result = data[data[column] > threshold]
    return result
