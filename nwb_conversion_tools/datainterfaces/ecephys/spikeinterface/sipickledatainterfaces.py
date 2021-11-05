"""Authors: Alessio Buccino."""
from spikeextractors import load_extractor_from_pickle

from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ..basesortingextractorinterface import BaseSortingExtractorInterface
from ....utils.json_schema import FilePathType
from ....utils import map_si_object_to_writer


class SIPickleRecordingExtractorInterface(BaseRecordingExtractorInterface):
    """Primary interface for reading and converting SpikeInterface Recording objects through .pkl files."""

    RX = None

    def __init__(self, pkl_file: FilePathType):
        self.recording_extractor = load_extractor_from_pickle(pkl_file)
        self.writer_class = map_si_object_to_writer(self.recording_extractor)(self.recording_extractor)
        self.subset_channels = None
        self.source_data = dict(pkl_file=pkl_file)


class SIPickleSortingExtractorInterface(BaseSortingExtractorInterface):
    """Primary interface for reading and converting SpikeInterface Sorting objects through .pkl files."""

    SX = None

    def __init__(self, pkl_file: FilePathType):
        self.sorting_extractor = load_extractor_from_pickle(pkl_file)
        self.writer_class = map_si_object_to_writer(self.sorting_extractor)(self.sorting_extractor)
        self.source_data = dict(pkl_file=pkl_file)
