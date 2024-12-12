"""TODO: Reference equinox and give credit."""

import abc
import dataclasses
import functools
import inspect
import warnings
import weakref
from typing import TYPE_CHECKING

from typing_extensions import dataclass_transform

_has_dataclass_init = weakref.WeakKeyDictionary()


@functools.wraps(dataclasses.field)
def field(*, static: bool = False, **kwargs):
    try:
        metadata = dict(kwargs.pop("metadata"))  # safety copy
    except KeyError:
        metadata = {}
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")

    if static:
        metadata["static"] = True
    return dataclasses.field(metadata=metadata, **kwargs)


def _is_abstract(cls):
    return len(cls.__abstractmethods__) > 0


class _ModuleMeta(abc.ABCMeta):
    # This method is called whenever you define a module: `class Foo(Module): ...`
    def __new__(mcs, name, bases, dict_, **kwargs):
        # Create class as normal (Equinox step 1).
        cls = super().__new__(mcs, name, bases, dict_, **kwargs)

        # Handle initialization (Equinox step 3).
        added_custom_init = "__init__" in cls.__dict__
        if added_custom_init:
            # The class has a custom `__init__` method.
            has_dataclass_init = False
        # Check if any super class has a custom `__init__` method. If they do, then we
        # just use that one. Note, this is different to default dataclass behaviour.
        # Given
        # ```
        # @dataclass
        # class Foo: def __init__(...): ...
        # @dataclass
        # class Bar(Foo): pass
        # ```
        # then `Bar` will end up with a dataclass-provided `__init__`. That ends up
        # being ergonomically very annoying, so we disable it.
        else:
            for kls in cls.__mro__[1:-1]:
                try:
                    has_dataclass_init = _has_dataclass_init[kls]
                except KeyError:
                    # Non-Module superclasses.
                    if kls.__init__ is not object.__init__:
                        has_dataclass_init = False
                        break
                else:
                    break
            else:
                assert name == "Module"
                has_dataclass_init = True  # eqx.Module itself

        if not has_dataclass_init and hasattr(cls, "__post_init__"):
            warnings.warn(
                f"Class `{cls.__module__}.{cls.__qualname__}` has both an "
                "`__init__` method and a `__post_init__` method. This means that "
                "the `__post_init__` method will not be run!\n"
                "The reason for this is that `__post_init__` is intended to be "
                "used with the automatically-generated `__init__` method provided "
                "by Python dataclasses, which are generated of the form:\n"
                "```\n"
                "def __init__(self, field1, field2)\n"
                "    self.field1 = field1\n"
                "    self.field2 = field2\n"
                "    self.__post_init__()\n"
                "```\n"
                "and as such a user-provided `__init__` overrides both the setting "
                "of fields, and the calling of `__post_init__`.\n",
                stacklevel=2,
            )

        # Fairly common to write `Superclass.__init__.__doc__ = "..."` with
        # dataclass-provided inits; here we look through the class hierarchy and will
        # copy this doc forward.
        if has_dataclass_init:
            init_doc = cls.__init__.__doc__

        # Register as a dataclass. (Equinox step 4).
        # Unlike Equinox, we don't use frozen dataclasses. This is a matter of
        # preference, and it is often easier to just mutate the object, especially when
        # experimenting interactively (like in a notebook setting).
        cls = dataclasses.dataclass(init=has_dataclass_init)(cls)

        # Registering here records that the `dataclass(...)` call has happened.
        _has_dataclass_init[cls] = has_dataclass_init

        # Assign `__doc__` in case it has been manually overriden:
        # ```
        # class Foo(Module):
        #     x: int
        #
        # Foo.__init__.__doc__ = "Foo should be called with with an integer `x`."
        #
        # class Bar(Foo):
        #     pass
        #
        # # Now we try to access `Bar.__init__.__doc__`. (E.g. during docgen.)
        # ```
        if has_dataclass_init:
            cls.__init__.__doc__ = init_doc  # pyright: ignore
            # TODO: is this next line still necessary?
            cls.__init__.__module__ = cls.__module__

        return cls

    @property
    def __signature__(cls):
        # Use signature of __init__ method for non-callable equinox modules
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.values())[1:]  # Remove self parameter
        return sig.replace(parameters=params)

    # This method is called whenever you initialise a module: `MyModule(...)`
    def __call__(cls, *args, **kwargs):
        # Instantiate the class as normal. (Equinox step 2)
        self = super(_ModuleMeta, cls).__call__(*args, **kwargs)
        assert not _is_abstract(cls)
        # Check that all fields are occupied. (Equinox step 3)
        missing_names = {
            field.name
            for field in dataclasses.fields(cls)  # pyright: ignore
            # Not `vars` or `__dict__`, to allow for `property`s overwriting a field.
            # Not recommended, but allowable for backward compatibility.
            if field.name not in dir(self)
        }
        if len(missing_names):
            raise ValueError(
                f"The following fields were not initialised during __init__: "
                f"{missing_names}"
            )

        return self

    # This bit is kind of sneaky. This is the reason that `_has_dataclass_init` is done
    # after dataclass registration -- we sneakily treat it as a marker for whether
    # dataclass registration has happened yet. And if it hasn't, we hide any
    # `__wrapped__` attribute. We want such attributes for the sake of
    # `module_update_wrapper`, but if `dataclass` sees it then it tries to follow it.
    def __getattribute__(cls, item):
        value = super().__getattribute__(item)
        if (
            item == "__wrapped__"
            and isinstance(value, property)
            and cls not in _has_dataclass_init
        ):
            raise AttributeError
        else:
            return value


if TYPE_CHECKING:

    @dataclass_transform(field_specifiers=(dataclasses.field, field))
    class _ModuleMeta(abc.ABCMeta): ...


def module_leaves(module: "Module") -> list:
    return [_field for _field in dataclasses.fields(module)]


class Module(metaclass=_ModuleMeta):
    # TODO: Re-add docstring from equinox
    def __hash__(self):
        return hash(tuple(module_leaves(self)))
