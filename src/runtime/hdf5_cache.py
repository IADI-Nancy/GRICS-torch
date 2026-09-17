"""Lossless HDF5 serialization for preprocessing arrays and their small metadata."""
import json
import h5py
import numpy as np


def write_tree(group, values):
    for key, value in values.items():
        name = str(key)
        if isinstance(value, dict):
            node = group.create_group(name)
            node.attrs['kind'] = 'dict'
            node.attrs['keys'] = json.dumps(list(value))
            write_tree(node, value)
        elif isinstance(value, (list, tuple)):
            node = group.create_group(name)
            node.attrs['kind'] = 'list'
            write_tree(node, dict(enumerate(value)))
        elif value is None:
            node = group.create_group(name)
            node.attrs['kind'] = 'none'
        elif isinstance(value, str):
            group.create_dataset(name, data=value, dtype=h5py.string_dtype('utf-8'))
        else:
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            group.create_dataset(name, data=value)


def read_tree(group):
    result = {}
    for name, node in group.items():
        if isinstance(node, h5py.Group):
            kind = node.attrs['kind']
            content = read_tree(node)
            if kind == 'none':
                value = None
            elif kind == 'list':
                value = [content[str(i)] for i in range(len(content))]
            else:
                value = {key: content[str(key)] for key in json.loads(node.attrs['keys'])}
        else:
            value = node.asstr()[()] if h5py.check_string_dtype(node.dtype) else node[()]
            if isinstance(value, np.generic):
                value = value.item()
        result[name] = value
    return result
