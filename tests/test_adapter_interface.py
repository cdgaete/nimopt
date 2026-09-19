import inspect

import pytest

from nimopt.solvers import ADAPTERS, adapter
from nimopt.solvers.base import Capabilities

SOLVE = ("assembled", "sense", "options")
CONFLICT = ("backend",)
RAY = ("backend",)


def names(held):
    """Return the parameter names of `held`, in order."""
    return tuple(inspect.signature(held).parameters)


@pytest.mark.parametrize("name", ADAPTERS)
def test_an_adapter_declares_a_backend_name_and_a_descriptor(name):
    held = adapter(name)
    assert isinstance(held.BACKEND, str)
    assert isinstance(held.CAPABILITIES, Capabilities)


@pytest.mark.parametrize("name", ADAPTERS)
def test_an_adapters_solve_takes_the_documented_parameters(name):
    held = adapter(name)
    assert callable(held.solve)
    assert names(held.solve) == SOLVE
    assert inspect.signature(held.solve).parameters["options"].default is None


@pytest.mark.parametrize("name", ADAPTERS)
def test_an_adapter_that_supports_conflict_takes_the_documented_parameters(name):
    held = adapter(name)
    if not held.CAPABILITIES.supports("conflict"):
        return
    assert callable(held.conflict)
    assert names(held.conflict) == CONFLICT


@pytest.mark.parametrize("name", ADAPTERS)
def test_an_adapter_that_supports_ray_takes_the_documented_parameters(name):
    held = adapter(name)
    if not held.CAPABILITIES.supports("ray"):
        return
    assert callable(held.ray)
    assert names(held.ray) == RAY
