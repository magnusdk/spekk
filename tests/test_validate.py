import numpy as np
import pytest
from spekk import Spec, ValidationError


def test_valid_data():
    Spec([]).validate([])
    Spec([["points"]]).validate([np.array([1, 2, 3])])
    Spec([["transmits", "receivers", "points"]]).validate([np.ones((3, 3, 4))])
    Spec([["points"], ["receivers"]]).validate(
        [np.array([1, 2, 3]), np.array([1, 2, 3])]
    )
    Spec([["transmits", "receivers", "points"], ["receivers"]]).validate(
        [np.ones((3, 3, 4)), np.array([1, 2, 3])]
    )
    Spec([[]]).validate(["arbitrary value"])
    Spec([[]]).validate([np.array(1)])
    Spec([["points"], []]).validate([np.array([1, 2, 3]), "arbitrary value"])
    Spec([["points"]], ["arg0"]).validate({"arg0": np.array([1, 2, 3])})


def test_invalid_data():
    with pytest.raises(ValidationError):
        Spec([["points"]]).validate([1])
    with pytest.raises(ValidationError):
        Spec([["points", "receivers"]]).validate([np.array([1, 2, 3])])
    with pytest.raises(ValidationError):
        Spec([[]]).validate([np.array([1, 2, 3])])
    with pytest.raises(ValueError):
        Spec([["points"]]).validate({"arg0": np.array([1, 2, 3])})
