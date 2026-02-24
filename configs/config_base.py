import inspect
import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, Field, fields, MISSING
from typing import List, Tuple, Union, Optional, AnyStr, Any
from typing import get_origin, get_args

from configs.serialization import Numpy2Json, to_json, from_data_file
from configs.types import DATA_DICT, correct_type

DATA_OBJ = Union[DATA_DICT, 'ConfigBase']


@dataclass
class ConfigBase(ABC):

    def __str__(self) -> AnyStr:
        s = f'\n[{self.__class__.__name__} -----'
        for key, value in sorted(self.__dict__.items()):
            s += f'\n o {key:45} | {value}'
        s += f'\n ----- {self.__class__.__name__}]'
        return s

    def as_dict(self, recursive: bool = True) -> DATA_DICT:
        dict_obj = self.__dict__.copy()
        if recursive:
            for key, value in dict_obj.items():
                if isinstance(value, ConfigBase):
                    dict_obj[key] = value.as_dict(recursive)
        return dict_obj

    def __getitem__(self, key: AnyStr) -> Any:
        return self.__dict__[key]

    @classmethod
    def get(cls, key: AnyStr) -> Any:
        return cls.__dict__.get(key)

    @classmethod
    def _from_dict(cls, dict_obj: DATA_DICT, raise_exception: bool = True) -> 'ConfigBase':
        # TODO: raise_exception backward compatibility must raise_exception = TRUE always
        missmatch_fields = list(set(dict_obj.keys()).difference(set(cls.fields_names())) &
                                set(cls.fields_names()).difference(set(dict_obj.keys())))
        if len(missmatch_fields) > 0:
            err_msg = f'{cls.__name__}: The following fields are missing: \n{missmatch_fields}'
            if raise_exception:
                raise ValueError(err_msg)
            else:
                warnings.warn(err_msg)

                # Best-effort coercion for primitive-typed fields before constructing dataclass
        for field in cls.get_fields():
            field_name = field.name
            if field_name in dict_obj:
                dict_obj[field_name] = cls._coerce_primitive(field.type, dict_obj[field_name])

        for field in cls.get_fields():
            try:
                field_name = field.name
                if cls.is_dataclass(field_name) and isinstance(dict_obj[field_name], dict):
                    dict_obj[field_name] = field.type.from_dict(dict_obj[field_name])
            except TypeError as err:
                # TODO backward compatibility: must raise exception
                warnings.warn(f'Field: {field_name}: {str(err)}')
        config = cls(**dict_obj)
        config.validate_type_fields(raise_exception)
        return config

    @staticmethod
    def _coerce_primitive(field_type: Any, value: Any) -> Any:
        """
        Best-effort coercion for config values coming from JSON/YAML/CLI.

        Only coerces a small set of safe primitives (and Optional[...] of those):
        float, int, bool, str.

        If coercion is not applicable or fails, returns the original value.
        """
        if value is None:
            return value

        origin = get_origin(field_type)
        args = get_args(field_type)

        # Optional[T] is Union[T, NoneType]
        if origin is Union and args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return ConfigBase._coerce_primitive(non_none[0], value)
            return value

        target = field_type
        if target is float and isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value

        if target is int and isinstance(value, str):
            try:
                # allow "10" but don't silently truncate "1e3"
                return int(value) if value.strip().isdigit() or (
                        value.strip().startswith(('-', '+')) and value.strip()[1:].isdigit()) else value
            except ValueError:
                return value

        if target is bool and isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "1", "yes", "y", "on"}:
                return True
            if v in {"false", "0", "no", "n", "off"}:
                return False
            return value

        if target is str and not isinstance(value, str):
            # Usually safe, but keep it conservative: only coerce basic primitives
            if isinstance(value, (int, float, bool)):
                return str(value)

        return value

    def save_to(self,
                file_path: str,
                indent: int = 4,
                encoder: json.JSONEncoder = Numpy2Json) -> str:
        return to_json(obj=self,
                       json_file=file_path,
                       indent=indent,
                       encoder=encoder)

    @classmethod
    def fields_names(cls) -> List[str]:
        return list(cls.__dataclass_fields__.keys())

    @classmethod
    def get_field(cls, name: str) -> Field:
        return cls.__dataclass_fields__[name]

    @classmethod
    def call_default_factory(cls, name: str) -> Optional[Any]:
        field_ = cls.get_field(name)
        if not field_.default_factory == MISSING:
            return field_.default_factory()
        return None

    @classmethod
    def get_fields(cls) -> Tuple[Field, ...]:
        return fields(cls)

    @classmethod
    def load_from(cls, data_obj: DATA_OBJ, raise_exception: bool = True) -> 'ConfigBase':
        if isinstance(data_obj, str):
            data_obj = from_data_file(data_obj)
        data_obj = cls._parse_data_obj(data_obj)
        data_obj.validate_type_fields(raise_exception)
        return data_obj

    @classmethod
    def _parse_data_obj(cls, data_obj: DATA_OBJ, raise_exception: bool = True) -> 'ConfigBase':
        # TODO: raise_exception backward compatibility must raise_exception = TRUE always
        if isinstance(data_obj, ConfigBase):
            data_obj = data_obj.__dict__
        elif not isinstance(data_obj, dict):
            raise TypeError(f'data_obj must dict of data class type: {type(data_obj)}')
        data_obj = cls._cleanup_not_valid_values(data_obj)
        for name, val in sorted(data_obj.items()):
            field_instance = cls.call_default_factory(name)
            if isinstance(val, ConfigBase):
                data_obj[name] = val._parse_data_obj(val, raise_exception=raise_exception)
            elif isinstance(val, dict) and isinstance(field_instance, ConfigBase):
                data_obj[name] = field_instance._parse_data_obj(val, raise_exception=raise_exception)
        data_obj = cls._from_dict(data_obj, raise_exception=raise_exception)
        return data_obj

    @classmethod
    def _cleanup_not_valid_values(cls, obj_dict: DATA_DICT) -> DATA_DICT:
        non_valid_keys = [key for key in obj_dict.keys() if not cls.exist(key)]
        if len(non_valid_keys) > 0:
            warnings.warn(f'Not valid name: {non_valid_keys}, will be removed from config dict')
            [obj_dict.pop(key_to_remove, None) for key_to_remove in non_valid_keys]
        return obj_dict

    @classmethod
    def load_config(cls,
                    config_obj: Union[str, dict, 'ConfigBase', None],
                    raise_exception: bool = True,
                    **kwargs) -> 'ConfigBase':
        # TODO added for backward compatibility,must be True always
        raise_exception = (len(kwargs) == 0) and raise_exception
        if config_obj is None:
            config = cls()
        elif isinstance(config_obj, (str, dict)):
            config = cls.load_from(config_obj, raise_exception)
        elif isinstance(config_obj, cls):
            config = cls._parse_data_obj(config_obj)
        else:
            raise TypeError(f'config_obj not valid type: {type(config_obj)}')
        config = config.update(**kwargs)
        config.validate_type_fields()
        return config

    def update(self, **kwargs) -> 'ConfigBase':
        """
        Updates the values of given configuration elements
        :param kwargs: The names and values of the configuration elements to update.
        :return: self
        """

        try:
            for key, value in kwargs.items():
                c = self.__dict__
                *sub_keys, final_key = key.split('.')
                for sub_key in sub_keys:
                    c = c[sub_key].__dict__
                c[final_key] = value
        except KeyError as e:
            raise KeyError(f'Non-valid update input {kwargs} , error message: {e}')

        return self

    def validate_type_fields(self, raise_exception: bool = True) -> None:
        """ Validates that each field in a dataclass instance conforms to its type annotation. """
        # TODO: raise_exception backward compatibility
        #  must raise_exception = TRUE always
        for field_ in self.get_fields():
            value_ = getattr(self, field_.name)
            if not correct_type(value_, field_.type):
                err_message = (f"Object: '{type(self)}', "
                               f"Field '{field_.name}', "
                               f"expected type {field_.type}, "
                               f"with value: {value_}, "
                               f"but got {type(value_).__name__}.")
                if raise_exception:
                    raise ValueError(str(err_message))
                else:
                    warnings.warn(f'{str(err_message)} \n '
                                  f'backward compatibility: must be raise_exception')
                    continue

            if isinstance(value_, ConfigBase):
                value_.validate_type_fields(raise_exception)

    @classmethod
    def is_field_class(cls, name: str) -> bool:
        field = cls.get_field(name)
        return inspect.isclass(field.type)

    @classmethod
    def is_dataclass(cls, field_name: str) -> bool:
        field = cls.get_field(field_name)
        return inspect.isclass(field.type) and issubclass(field.type, ConfigBase)

    @classmethod
    def exist(cls, field_name: str) -> bool:
        return field_name in cls.fields_names()


@dataclass
class ParamsBase(ConfigBase, ABC):
    # validate_on_init: bool = field(default=True, repr=False, metadata={"serialize": False})

    @abstractmethod
    def check_valid(self, **kwargs) -> None:
        raise NotImplementedError

    def __post_init__(self):
        pass
        # if self.validate_on_init:
        #     self.check_valid()
