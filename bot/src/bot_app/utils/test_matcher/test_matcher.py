import pytest
import numpy as np

from matcher.matcher import Matcher

FIXTURES_PATH = "test_matcher/fixtures"
TEST_PATH = "test_matcher/test_files"


@pytest.mark.parametrize(
    "args",
    [
        (1, 2, 3, 4, 5),
        (2.1, 2.5, f"{FIXTURES_PATH}/faulty.bin"),
        ("asf", "2", "\\directory", 2, 5),
        (2, 5, None, None, None),
        (2, 5, None, 5, 1.2),
    ],
)
def test_incorrect_input_type(args):
    with pytest.raises(TypeError):
        _ = Matcher(*args)


def test_incorrect_input_path():
    with pytest.raises(RuntimeError):
        _ = Matcher(2, 5, path_to_index=f"{FIXTURES_PATH}/faulty.bin")


def test_incorrect_input():
    with pytest.raises(RuntimeError):
        _ = Matcher(2, 5, path_to_index=f"{FIXTURES_PATH}/trash")


def test_saving_index():
    correctly_saved_path = f"{FIXTURES_PATH}/saved_index.bin"
    test_saved_path = f"{TEST_PATH}/test_saved_index.bin"

    matcher = Matcher(2, 5, M=42, ef_construction=5)
    matcher.add_items(np.array([[1, 2], [5, 6], [7, 8]]))
    matcher.save_index(test_saved_path)

    with open(correctly_saved_path, "rb") as correct_result, open(
        test_saved_path, "rb"
    ) as test_result:
        assert correct_result.read() == test_result.read()


@pytest.mark.parametrize(
    "data, exception",
    [
        (5, TypeError),
        ("something", TypeError),
        (np.arange(10), ValueError),
        (np.arange(8).reshape((2, 2, 2)), ValueError),
    ],
)
def test_nn_search_incorrect_input(data, exception):
    matcher = Matcher(10, 10, M=46, ef_construction=16)
    matcher.add_items(np.arange(10 * 10).reshape((10, 10)))

    with pytest.raises(exception):
        matcher.get_nearest_neighbour(data)


# the algorithm is apporxiamte,
# so the tests are given just to check its
# results on small datasets
@pytest.mark.parametrize(
    "dim, max_elements",
    [
        (2, 10),
        (2, 100),
        (2, 1000),
        (10, 10),
        (10, 100),
        (10, 1000),
        (1000, 10),
        (1000, 100),
        (1000, 1000),
    ],
)
def test_nn_search(dim: int, max_elements: int):
    path = f"{TEST_PATH}/test_saved_index.bin"

    to_save = np.arange(dim * max_elements).reshape((max_elements, dim))
    saving_matcher = Matcher(dim, max_elements, M=46, ef_construction=16)
    saving_matcher.add_items(to_save)
    saving_matcher.save_index(path)

    matcher = Matcher(dim, max_elements, path_to_index=path)
    matcher.get_nearest_neighbour(np.expand_dims(to_save[5], axis=0))
