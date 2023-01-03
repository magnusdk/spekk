from spekk import Spec


def test_replace_with_nested_spec_regression():
    spec = Spec({"a": ["d1", "d2", "d3"]})
    new_spec = spec.replace({"a": Spec(["d4", "d5"])})
    assert new_spec == Spec({"a": ["d4", "d5"]})


def test_replace_with_new_keys_regression():
    spec = Spec({"a": ["d1", "d2", "d3"]})
    new_spec = spec.replace({"b": Spec(["d4"])})
    assert new_spec == Spec({"a": ["d1", "d2", "d3"], "b": ["d4"]})

    spec = Spec({"a": {"b": ["d1", "d2", "d3"]}})
    new_spec = spec.replace({"a": {"c": Spec(["d4"])}})
    assert new_spec == Spec({"a": {"b": ["d1", "d2", "d3"], "c": ["d4"]}})
