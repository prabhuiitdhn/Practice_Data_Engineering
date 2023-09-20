# https://gitlab.com/marvelousmlops/cicd-for-mlops/-/tree/part1_setting-up-your-first-ci-pipeline?ref_type=heads

from src.preprocessing import filter_data
from pandas import DataFrame


def test_filter_data():
    test_data = DataFrame({"column_a": [50, 150, 250, 300], "column_b": [4, 7, 3, 5]})
    expected_result = DataFrame({"column_a": [150, 250, 300], "column_b": [7, 3, 5]})
    result = filter_data(test_data, "column_a", 100).reset_index(drop=True)
    assert result.equals(expected_result)
