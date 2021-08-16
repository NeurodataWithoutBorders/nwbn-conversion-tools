import uuid
from datetime import datetime
import warnings
import numpy as np
import distutils.version
from pathlib import Path
from typing import Union, Optional, List
from distutils.version import StrictVersion
from warnings import warn
import psutil
from collections import defaultdict

import pynwb
from numbers import Real
from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5.h5_utils import H5DataIO
from .json_schema import dict_deep_update
from .basenwbephyswriter import BaseNwbEphysWriter
from .common_writer_tools import ArrayType, PathType, set_dynamic_table_property, check_module, list_get

try:
    import neo

    if StrictVersion(neo.__version__) >= StrictVersion("0.10"):
        HAVE_NEO = True
    else:
        HAVE_NEO = False
except ImportError:
    HAVE_NEO = False


class NEONwbEphysWriter(BaseNwbEphysWriter):
    """
    Class to write a neo.RawIO or neo.IO versio>=0.10 to NWB

    Parameters
    ----------
    object_to_write: neo.RawIO or neo.IO
    nwb_file_path: Path type
    nwbfile: pynwb.NWBFile or None
    metadata: dict or None
    **kwargs: list kwargs and meaning
    """

    def __init__(
        self,
        object_to_write: Union[neo.rawio.baserawio.BaseRawIO, neo.io.baseio.BaseIO],
        nwb_file_path: PathType = None,
        nwbfile: pynwb.NWBFile = None,
        metadata: dict = None,
        **kwargs,
    ):
        assert HAVE_NEO
        self.recording, self.sorting, self.event = None, None, None
        BaseNwbEphysWriter.__init__(
            self, object_to_write, nwb_file_path=nwb_file_path, nwbfile=nwbfile, metadata=metadata, **kwargs
        )

    @staticmethod
    def supported_types():
        assert HAVE_NEO
        return (neo.rawio.baserawio.BaseRawIO, neo.io.baseio.BaseIO)

    def write_to_nwb(self):
        # check what's in the neo object: analogsignals, spike trains, events and
        # write recording, sorting, events accordingly
        pass

    def write_recording(self):
        raise NotImplementedError

    def write_sorting(self):
        raise NotImplementedError

    def write_epochs(self):
        raise NotImplementedError

    def get_nwb_metadata(self):
        raise NotImplementedError

    def add_electrodes(self):
        raise NotImplementedError

    def add_electrode_groups(self):
        raise NotImplementedError

    def add_electrical_series(self):
        raise NotImplementedError

    def add_units(self):
        raise NotImplementedError

    def add_epochs(self):
        raise NotImplementedError
