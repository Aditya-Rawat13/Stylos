"""
Simple validation tests to verify test infrastructure.
These tests don't require complex dependencies.
"""
import pytest


def test_basic_math():
    """Test basic arithmetic to verify pytest works."""
    assert 1 + 1 == 2
    assert 10 - 5 == 5
    assert 3 * 4 == 12


def test_string_operations():
    """Test string operations."""
    text = "Hello World"
    assert text.lower() == "hello world"
    assert text.upper() == "HELLO WORLD"
    assert len(text) == 11


def test_list_operations():
    """Test list operations."""
    items = [1, 2, 3, 4, 5]
    assert len(items) == 5
    assert sum(items) == 15
    assert max(items) == 5
    assert min(items) == 1


def test_dict_operations():
    """Test dictionary operations."""
    data = {"name": "Test", "value": 42}
    assert data["name"] == "Test"
    assert data["value"] == 42
    assert "name" in data
    assert "missing" not in data


class TestClassExample:
    """Example test class."""
    
    def test_method_one(self):
        """Test method one."""
        assert True
    
    def test_method_two(self):
        """Test method two."""
        result = [x * 2 for x in range(5)]
        assert result == [0, 2, 4, 6, 8]


@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
])
def test_parametrized(input, expected):
    """Test parametrized test."""
    assert input * 2 == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
