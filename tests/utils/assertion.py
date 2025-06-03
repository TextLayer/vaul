from typing import Any, Optional, Sequence, Type


def is_equal(
    value: Any, expected_value: Any, error: str = "The two values are not equal"
) -> None:
    """Assert that two values are equal.

    Args:
        value: First value to compare
        expected_value: Second value to compare
        error: Custom error message if assertion fails
    """
    assert value == expected_value, error


def is_not_equal(
    value: Any, expected_value: Any, error: str = "The two values are equal"
) -> None:
    """Assert that two values are not equal.

    Args:
        value: First value to compare
        expected_value: Second value to compare
        error: Custom error message if assertion fails
    """
    assert value != expected_value, error


def is_none(value: Any, error: str = "Value is not None") -> None:
    """Assert that a value is None.

    Args:
        value: Value to check
        error: Custom error message if assertion fails
    """
    assert value is None, error


def is_not_none(value: Any, error: str = "Value is not None") -> None:
    """Assert that a value is not None.

    Args:
        value: Value to check
        error: Custom error message if assertion fails
    """
    assert value is not None, error


def is_true(value: bool, error: str = "Value is not True") -> None:
    """Assert that a value is True.

    Args:
        value: Value to check
        error: Custom error message if assertion fails
    """
    assert value is True, error


def is_false(value: bool, error: str = "Value is not False") -> None:
    """Assert that a value is False.

    Args:
        value: Value to check
        error: Custom error message if assertion fails
    """
    assert value is False, error


def contains(sequence: Sequence, item: Any, error: Optional[str] = None) -> None:
    """Assert that a sequence contains an item.

    Args:
        sequence: Sequence to check
        item: Item to look for
        error: Custom error message if assertion fails
    """
    error = error or f"Sequence does not contain {item}"
    assert item in sequence, error


def not_contains(sequence: Sequence, item: Any, error: Optional[str] = None) -> None:
    """Assert that a sequence does not contain an item.

    Args:
        sequence: Sequence to check
        item: Item to look for
        error: Custom error message if assertion fails
    """
    error = error or f"Sequence contains {item}"
    assert item not in sequence, error


def is_instance(value: Any, expected_type: Type, error: Optional[str] = None) -> None:
    """Assert that a value is an instance of the expected type.

    Args:
        value: Value to check
        expected_type: Expected type of the value
        error: Custom error message if assertion fails
    """
    error = error or f"Value {value} is not an instance of {expected_type}"
    assert isinstance(value, expected_type), error
