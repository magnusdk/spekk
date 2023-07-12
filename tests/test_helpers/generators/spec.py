from hypothesis import strategies as st

from spekk import Spec

alphabet = "abcdefghijklmnopqrstuvwxyz"
keys = st.text(alphabet, min_size=1, max_size=3)
dimensions = st.lists(keys, min_size=0, max_size=5, unique=True)


def _specs_tree():
    return st.recursive(
        dimensions,
        lambda children: st.one_of(
            st.lists(children, max_size=3),
            st.dictionaries(keys, children, max_size=3),
        ),
        max_leaves=10,
    )


def specs():
    return _specs_tree().map(Spec)


def kwargs_specs():
    return st.dictionaries(keys, _specs_tree(), max_size=3).map(Spec)
