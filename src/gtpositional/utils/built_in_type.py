import typing
from numbers import Number
from typing import Literal, Tuple

def _type_convert(param):
    if isinstance(param, Number):
        return Literal[param]
    elif isinstance(param, typing.Tuple):
        return Tuple[tuple(_type_convert(el) for el in param)]
    return param

def built_in_type(definition):
    """
    ```
    Decorator for built-in-types to allow passing numbers and tuples like TypeVars

    .. code-block:: python
        @built_in_type
        class SomeBuiltInType(Generic[T1, T2]):
            pass

        SomeBuiltInType[1, (1, 1)]
    :param definition:
    :return:
    """
    @classmethod
    def class_getitem(cls, params):
        if not isinstance(params, tuple):
            params = (params,)

        new_params = tuple(map(_type_convert, params))

        return super(definition, cls).__class_getitem__(new_params)

    setattr(definition, "__class_getitem__", class_getitem)

    return definition