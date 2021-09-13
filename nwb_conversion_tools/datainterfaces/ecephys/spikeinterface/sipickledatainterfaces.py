"""Authors: Alessio Buccino."""
from spikeextractors import load_extractor_from_pickle

from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ..basesortingextractorinterface import BaseSortingExtractorInterface
from ....utils.json_schema import FilePathType


class SIPickleRecordingExtractorInterface(BaseRecordingExtractorInterface):
    """Primary interface for reading and converting SpikeInterface objects through Pickle files."""

    RX = load_extractor_from_pickle

    def __init__(self, file_path: FilePathType):
        super().__init__(pkl_file=file_path)


class SIPickleSortingExtractorInterface(BaseSortingExtractorInterface):
    """Primary interface for reading and converting SpikeInterface objects through Pickle files."""

    SX = load_extractor_from_pickle

    def __init__(self, file_path: FilePathType):
        super().__init__(pkl_file=file_path)
