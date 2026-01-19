from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from spekk import Module, flatten


class A(Module):
    subtree1: Any
    subtree2: Any


class B(Module):
    subtree1: Any
    subtree2: Any


def nested_structures(
    *,
    max_leaves: int = 80,
    max_container_size: int = 6,
    max_dict_size: int = 6,
):
    leaf = st.floats()
    dict_key = st.text(min_size=0, max_size=20)

    # Build recursive containers + A/B nodes
    return st.recursive(
        base=leaf,
        extend=lambda inner: st.one_of(
            st.lists(inner, min_size=0, max_size=max_container_size),
            st.tuples(inner, inner),
            st.dictionaries(dict_key, inner, min_size=0, max_size=max_dict_size),
            st.builds(A, inner, inner),
            st.builds(B, inner, inner),
        ),
        max_leaves=max_leaves,
    )


@settings(max_examples=100, deadline=None)
@given(obj=nested_structures())
def test_flatten_unflatten(obj):
    dynamic, treedef = flatten(obj)
    out = treedef.unflatten(dynamic)
    assert out == obj
