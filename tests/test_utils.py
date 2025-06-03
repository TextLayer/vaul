import pytest
from vaul import utils
from tests.utils.assertion import is_equal, is_not_none


def test_import():
    """Test that the utils module can be imported."""
    is_not_none(utils)


def test_remove_keys_recursively_single_key():
    """Test removing a single key from a dictionary."""
    input_dict = {"a": 1, "b": 2, "c": 3}
    result = utils.remove_keys_recursively(input_dict, "b")
    is_equal(result, {"a": 1, "c": 3})


def test_remove_keys_recursively_multiple_keys():
    """Test removing multiple keys from a dictionary."""
    input_dict = {"a": 1, "b": 2, "c": 3}
    result = utils.remove_keys_recursively(input_dict, ["a", "c"])
    is_equal(result, {"b": 2})


def test_remove_keys_recursively_nested_dict():
    """Test removing keys from a nested dictionary."""
    input_dict = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
    result = utils.remove_keys_recursively(input_dict, "c")
    is_equal(result, {"a": 1, "b": {"d": 3}, "e": 4})


def test_remove_keys_recursively_type_error():
    """Test that TypeError is raised when input is not a dictionary."""
    with pytest.raises(TypeError):
        utils.remove_keys_recursively("not a dict", "key")
