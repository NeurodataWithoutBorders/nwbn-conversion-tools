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
from .basesinwbephyswriter import BaseSINwbEphysWriter
from .common_writer_tools import ArrayType, PathType, set_dynamic_table_property, check_module, list_get

try:
    import spikeinterface as si

    if StrictVersion(si.__version__[:4]) >= StrictVersion("0.90"):
        HAVE_SI_090 = True
    else:
        HAVE_SI_090 = False
except ImportError:
    HAVE_SI_090 = False


class SI090NwbEphysWriter(BaseSINwbEphysWriter):
    """
    Class to write a Recording, Sorting, Event, or WaveformExtractor object from SI>=0.90 to NWB

    Parameters
    ----------
    object_to_write: si.BaseRecording, si.BaseSorting, si.BaseEvent, si.WaveformExtractor
    nwb_file_path: Path type
    nwbfile: pynwb.NWBFile or None
    metadata: dict or None
    **kwargs: list kwargs and meaning
    """

    def __init__(
        self,
        object_to_write: Union[si.BaseRecording, si.BaseSorting, si.BaseEvent, si.WaveformExtractor],
        nwb_file_path: PathType = None,
        nwbfile: pynwb.NWBFile = None,
        metadata: dict = None,
        **kwargs,
    ):
        assert HAVE_SI_090
        BaseSINwbEphysWriter.__init__(
            self, object_to_write, nwb_file_path=nwb_file_path, nwbfile=nwbfile, metadata=metadata, **kwargs
        )

    @staticmethod
    def supported_types():
        assert HAVE_SI_090
        return (si.BaseRecording, si.BaseSorting, si.BaseEvent, si.WaveformExtractor)

    def add_to_nwb(self):
        if isinstance(self.object_to_write, si.BaseRecording):
            self.recording = self.object_to_write
            self.add_recording()
        elif isinstance(self.object_to_write, si.BaseRecording):
            self.sorting = self.object_to_write
            self.add_sorting()
        elif isinstance(self.object_to_write, si.BaseEvent):
            self.event = self.object_to_write
            self.add_epochs()
        elif isinstance(self.object_to_write, si.WaveformExtractor):
            self.recording = self.object_to_write.recording
            self.sorting = self.object_to_write.sorting
            self.waveforms = self.object_to_write
            self.add_recording()
            self.add_sorting()
            self.add_waveforms()

    def _get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        return self.recording.get_traces(channel_ids=None, start_frame=None, end_frame=None, return_scaled=True).T

    def _get_channel_property_names(self, chan_id):
        return self.recording.get_property_keys()

    def _get_channel_property_values(self, prop, chan_id):
        if prop == "location":
            try:
                return self.recording.get_channel_locations(channel_ids=chan_id)
            except:
                return np.nan * np.ones(len(self._get_channel_ids()), 2)
        elif prop == "gain":
            if self.recording.get_channel_gains(channel_ids=chan_id) is None:
                return np.ones(len(self._get_channel_ids()))
            else:
                return self.recording.get_channel_gains(channel_ids=chan_id)
        elif prop == "offset":
            if self.recording.get_channel_offsets(channel_ids=chan_id) is None:
                return np.zeros(len(self._get_channel_ids()))
            else:
                return self.recording.get_channel_offsets(channel_ids=chan_id)
        elif prop == "group":
            if self.recording.get_channel_groups(channel_ids=chan_id) is None:
                return np.zeros(len(self._get_channel_ids()))
            else:
                return self.recording.get_channel_groups(channel_ids=chan_id)
        self.recording.get_property(prop, ids=chan_id)

    def _get_times(self):
        return np.range(0, self._get_num_frames() * self._get_sampling_frequency(), self._get_sampling_frequency())

    def _get_unit_feature_names(self, unit_id):
        return []

    def _get_unit_feature_values(self, prop, unit_id):
        return

    def _get_unit_spike_train_times(self, unit_id):
        return self._get_unit_spike_train_ids(unit_id) / self._get_unit_sampling_frequency()

    def _get_unit_property_names(self, unit_id):
        return list(self.sorting._properties.keys())

    def _get_unit_property_values(self, prop, unit_id):
        return self.sorting.get_unit_property(unit_id, prop)

    def add_epochs(self):
        raise NotImplementedError

    def add_waveforms(self):
        raise NotImplementedError

    def add_epochs(self):
        raise NotImplementedError
