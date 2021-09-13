"""Authors: Cody Baker."""
from spikeextractors import SpikeGadgetsRecordingExtractor, load_probe_file

from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ....utils.json_schema import FilePathType, OptionalFilePathType


class SpikeGadgetsRecordingInterface(BaseRecordingExtractorInterface):
    """Primary data interface class for converting the SpikeGadgets format."""

    RX = SpikeGadgetsRecordingExtractor

    @classmethod
    def get_source_schema(cls):
        source_schema = super().get_source_schema()
        source_schema["properties"]["file_path"].update(description="Path to SpikeGadgets (.rec) file.")
        source_schema["properties"]["probe_file_path"].update(
            description="Optional path to a probe (.prb) file describing electrode features."
        )
        return source_schema

    def __init__(self, file_path: FilePathType, probe_file_path: OptionalFilePathType = None):
        super().__init__(filename=file_path)
        if probe_file_path is not None:
            self.recording_extractor = load_probe_file(recording=self.recording_extractor, probe_file=probe_file_path)
