import uuid
from datetime import datetime
import warnings
import numpy as np
import distutils.version
from pathlib import Path
from typing import Union, Optional, List
from warnings import warn
import psutil
from collections import defaultdict
from copy import deepcopy

import pynwb
from numbers import Real
from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5.h5_utils import H5DataIO
from .json_schema import dict_deep_update
from .basenwbephyswriter import BaseNwbEphysWriter
from .basesinwbephyswriter import BaseSINwbEphysWriter
from .common_writer_tools import ArrayType, PathType, set_dynamic_table_property, check_module, list_get

try:
    import spikeextractors as se

    HAVE_SI013 = True
except ImportError:
    HAVE_SI013 = False


class SI013NwbEphysWriter(BaseSINwbEphysWriter):
    """
    Class to write RecordingExtractor and SortingExtractor object from SI<=0.13 to NWB

    Parameters
    ----------
    object_to_write: se.RecordingExtractor or se.SortingExtractor
    nwb_file_path: Path type
    nwbfile: pynwb.NWBFile or None
    metadata: dict or None
    **kwargs: list kwargs and meaning
    """

    def __init__(
        self,
        object_to_write: Union[se.RecordingExtractor, se.SortingExtractor],
        nwbfile: pynwb.NWBFile = None,
        metadata: dict = None,
        **kwargs,
    ):
        assert HAVE_SI013, "spikeextractors 0.13 version not installed"
        BaseSINwbEphysWriter.__init__(self, object_to_write, nwbfile=nwbfile, metadata=metadata, **kwargs)
        if isinstance(self.object_to_write, se.RecordingExtractor):
            self.recording = self.object_to_write
        elif isinstance(self.object_to_write, se.SortingExtractor):
            self.sorting = self.object_to_write

    @staticmethod
    def supported_types():
        assert HAVE_SI013
        return (se.RecordingExtractor, se.SortingExtractor)

    def add_to_nwb(self):
        if isinstance(self.object_to_write, se.RecordingExtractor):
            self.add_recording()
        elif isinstance(self.object_to_write, se.SortingExtractor):
            self.add_sorting()

    def _get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        return self.recording.get_traces(channel_ids=None, start_frame=None, end_frame=None, return_scaled=True)

    def _get_channel_property_names(self, chan_id):
        return self.recording.get_channel_property_names(channel_id=chan_id)

    def _get_channel_property_values(self, prop, chan_id):
        if prop == "location":
            return self.recording.get_channel_locations(channel_ids=chan_id)
        elif prop == "gain":
            return self.recording.get_channel_gains(channel_ids=chan_id)
        elif prop == "offset":
            return self.recording.get_channel_offsets(channel_ids=chan_id)
        elif prop == "group":
            return self.recording.get_channel_groups(channel_ids=chan_id)
        return self.recording.get_channel_property(channel_id=chan_id, property_name=prop)

    def _get_times(self):
        if self.recording._times is None:
            return np.range(0, self._get_num_frames() * self._get_sampling_frequency(), self._get_sampling_frequency())
        return self.recording._times

    def _get_unit_feature_names(self, unit_id):
        return self.sorting.get_unit_spike_feature_names(unit_id)

    def _get_unit_feature_values(self, prop, unit_id):
        return self.sorting.get_unit_spike_features(unit_id, prop)

    def _get_unit_spike_train_times(self, unit_id):
        return self.sorting.frame_to_time(self.sorting.get_unit_spike_train(unit_id=unit_id))

    def _get_unit_property_names(self, unit_id):
        return self.sorting.get_unit_property_names(unit_id)

    def _get_unit_property_values(self, prop, unit_id):
        return self.sorting.get_unit_property(unit_id, prop)

    def add_epochs(self):
        """
        Auxiliary static method for nwbextractor.

        Adds epochs from recording object to nwbfile object.

        """
        if self.nwbfile is not None:
            assert isinstance(self.nwbfile, pynwb.NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

        # add/update epochs
        for epoch_name in self.recording.get_epoch_names():
            epoch = self.recording.get_epoch_info(epoch_name)
            if self.nwbfile.epochs is None:
                self.nwbfile.add_epoch(
                    start_time=self.recording.frame_to_time(epoch["start_frame"]),
                    stop_time=self.recording.frame_to_time(epoch["end_frame"] - 1),
                    tags=epoch_name,
                )
            else:
                if [epoch_name] in self.nwbfile.epochs["tags"][:]:
                    ind = self.nwbfile.epochs["tags"][:].index([epoch_name])
                    self.nwbfile.epochs["start_time"].data[ind] = self.recording.frame_to_time(epoch["start_frame"])
                    self.nwbfile.epochs["stop_time"].data[ind] = self.recording.frame_to_time(epoch["end_frame"])
                else:
                    self.nwbfile.add_epoch(
                        start_time=self.recording.frame_to_time(epoch["start_frame"]),
                        stop_time=self.recording.frame_to_time(epoch["end_frame"]),
                        tags=epoch_name,
                    )
