import inspect
from functools import wraps
from typing import TypeVar, List, Optional, \
    AnyStr, Dict, Any, Callable, \
    get_type_hints, Type, Union, Sequence, \
    MutableSequence, Tuple, MutableMapping, \
    get_origin, get_args

import numpy as np

RT = TypeVar('RT')  # return type
PathStr = TypeVar('PathStr', bound=str)
PathList = List[PathStr]
OPT_STR = Optional[AnyStr]
DATA_DICT = Dict[str, Any]
PARAMS_DICT = Dict[str, AnyStr]
PARAMS_LIST = List[AnyStr]
NUMERIC_TYPES_ = (int, float, complex)
NUMBER_TYPES = Union[NUMERIC_TYPES_]
PRIMITIVE_TYPES = (int, float, bool, complex, str)


def args_type_check(func: Callable[..., RT]) -> Callable[..., RT]:
    """
    Decorator that checks if all input arguments of a function are of the correct types
   for example def my_func(arg1: bool, arg2: str) will raise an error if in calling
   in cli 'python my_func --args false args2 name', because must be '--args False'
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> RT:
        # Get type hints from the function
        type_hints = get_type_hints(func)

        # Get the function's signature
        sig = inspect.signature(func)

        # Bind the arguments to the signature
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Check the types of the arguments
        for name, value in bound_args.arguments.items():
            if name in type_hints:
                expected_type = type_hints[name]
                if not correct_type(value, expected_type):
                    raise TypeError(f"{func}: Argument '{name}' must be {expected_type}, "
                                    f"but got value {value} is {type(value)}")

        return func(*args, **kwargs)

    return wrapper


def is_class_type(type_hint: Type) -> bool:
    return inspect.isclass(type_hint)


def np_to_py_scalar(var: Any) -> Union[PRIMITIVE_TYPES]:
    if isinstance(var, np.generic) and np.isscalar(var):
        return var.item()
    return var


def is_primitive_type(type_hint: Type) -> bool:
    return is_class_type(type_hint) and issubclass(np_to_py_scalar(type_hint), PRIMITIVE_TYPES)


def is_numeric_type(type_hint: Type) -> bool:
    return is_class_type(type_hint) and issubclass(type_hint, NUMERIC_TYPES_)


def is_type_sequence_class(type_hint: Type) -> bool:
    return is_class_type(type_hint) and issubclass(type_hint, (Sequence, MutableSequence, List, Tuple))


def is_type_dict_class(type_hint: Type) -> bool:
    return is_class_type(type_hint) and issubclass(type_hint, (MutableMapping, dict, Dict))


def get_args_type(type_hint: Type) -> List[Type]:
    args = list(get_args(type_hint))
    if isinstance(type_hint, TypeVar):
        args += [type_hint.__bound__] + list(type_hint.__constraints__)
    return args


def correct_type(var: Any, type_hint: Type) -> bool:
    """
    Check if the given value conforms to the specified type_hint.
    Handles List, Optional, Union, Sequence, and Dict from typing module.
    """
    origin_type = get_origin(type_hint) or type_hint
    args = get_args_type(type_hint)

    if origin_type is Any:
        return True

    if is_primitive_type(origin_type) and np.isscalar(var):
        return isinstance(np_to_py_scalar(var), origin_type)

    if origin_type is Union or isinstance(origin_type, TypeVar):
        # Handle Optional (as it is Union[X, None]) and Union types
        return any(correct_type(var, arg) for arg in args)
    elif is_type_sequence_class(origin_type):
        # Check for list and sequences
        if not isinstance(var, (list, tuple, np.ndarray)):
            return False
        element_type = args[0]
        if isinstance(var, np.ndarray):
            var = var.tolist()
        return all(correct_type(item, element_type) for item in var)
    elif is_type_dict_class(origin_type):
        # Check for dict and mappings
        if not isinstance(var, dict):
            return False
        key_type, value_type = args
        return all(correct_type(k, key_type) and correct_type(v, value_type) for k, v in var.items())
    elif is_class_type(origin_type):
        # Default case for simple types and any other directly checkable types
        return isinstance(var, origin_type)
    else:
        return False


def ensure_args_str(func: Callable[..., RT]) -> Callable[..., RT]:
    def wrapper(cls, *args: str, **kwargs: Dict[str, str]) -> RT:
        # The first argument is cls, which we do not want to convert
        convert_to_str = lambda x: str(x) if isinstance(x, (int, float, bytes)) else x
        new_args = [convert_to_str(arg) for arg in args]
        new_kwargs = {k: convert_to_str(v) for k, v in kwargs.items()}
        return func(cls, *new_args, **new_kwargs)

    return wrapper


def ensure_args_path(func: Callable[..., RT]) -> Callable[..., RT]:
    return ensure_args_str(func)


def get_var_name(var: Any) -> AnyStr:
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [name for name, val in callers_local_vars if val is var][0]
