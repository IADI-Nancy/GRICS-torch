"""Read-only access to shared ISMRMRD inputs and their cached XML headers."""
from pathlib import Path

import h5py
import ismrmrd


class ReadOnlyDataset(ismrmrd.Dataset):
    """Use the ISMRMRD reading API without its default r+ HDF5 open mode.

    The installed ISMRMRD Dataset constructor has no read-only option. Its read
    methods use these two backing attributes, which we initialize explicitly.
    """
    def __init__(self, filename, dataset_name='dataset'):
        self._file = h5py.File(filename, 'r')
        self._dataset_name = dataset_name


def read_header(path: str | Path):
    with h5py.File(path, 'r') as handle:
        xml = handle['dataset/xml'][0]
    return ismrmrd.xsd.CreateFromDocument(xml)


def acquisition_header(raw_data=None, ismrmrd_file=None):
    """An explicitly supplied file wins; otherwise prefer the in-memory header."""
    if ismrmrd_file is not None:
        return read_header(ismrmrd_file)
    if raw_data is not None:
        xml = getattr(raw_data, 'ismrmrd_header', None)
        if xml is not None:
            return ismrmrd.xsd.CreateFromDocument(xml)
        for name in ('source_ismrmrd_file', 'ismrmrd_file'):
            path = getattr(raw_data, name, None)
            if path:
                return read_header(path)
        filenames = getattr(raw_data, 'rawdata_filenames', None)
        if filenames:
            return read_header(filenames[0])
    raise ValueError('Provide raw_data with an ISMRMRD header or ismrmrd_file=...')
