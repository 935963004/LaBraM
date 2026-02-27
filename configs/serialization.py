import json
import os
from typing import Any

import jsonpickle
import numpy as np
import yaml


class Numpy2Json(json.JSONEncoder):
    """
    A JSONEncoder capable of converting numpy types to simple python builtin types.
    """

    # pylint: disable=method-hidden
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.int64, np.int32, np.int16)):
            return int(obj)

        return json.JSONEncoder.default(self, obj)


def from_data_file(file_url: str) -> Any:
    if file_url.endswith('.json'):
        return from_json(file_url)
    elif file_url.endswith('.yaml'):
        return from_yaml(file_url)
    else:
        raise TypeError(f'Non-supported format of file: {file_url}. '
                        f'Supported formats are JSON or YAML.')


def to_data_file(obj: Any, fil_url: str, indent: int = 4, encoder: json.JSONEncoder = Numpy2Json) -> str:
    if fil_url.endswith('.json'):
        return to_json(obj, fil_url, indent, encoder)
    elif fil_url.endswith('.yaml'):
        return to_yaml(obj, fil_url)
    else:
        raise TypeError(f'Non-supported format of file: {fil_url}. '
                        f'Supported formats are JSON or YAML.')


def from_json(json_file: str) -> Any:
    if not os.path.isfile(json_file):
        raise FileExistsError(f'Not exist json_file: {json_file}')
    with open(json_file, 'r') as f:
        return jsonpickle.decode(f.read())


def to_json(obj: Any, json_file: str, indent: int = 4, encoder: json.JSONEncoder = Numpy2Json) -> str:
    json_str = jsonpickle.encode(obj)
    # load the encoded JSON so we can save it with in a pretty
    # format with indentations (jsonpickle does not support this)
    json_dict = json.loads(json_str)
    with open(json_file, 'w') as f:
        json.dump(json_dict, f, indent=indent, cls=encoder)
    if not os.path.isfile(json_file):
        raise RuntimeError(f'Cannot save to json file: {json_file}')
    return json_file


def from_yaml(yaml_url: str) -> Any:
    if not os.path.isfile(yaml_url):
        raise FileNotFoundError(f'Not exist yaml file: {yaml_url}')
    with open(yaml_url, 'r', encoding='utf-8') as f:
        obj = yaml.safe_load(f)
    return obj


def to_yaml(obj: Any, output_path: str)-> str:
    with open(output_path, 'w') as f:
        yaml.dump(obj, f)
        return output_path

