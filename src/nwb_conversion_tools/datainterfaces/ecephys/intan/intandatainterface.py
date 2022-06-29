"""Authors: Cody Baker and Ben Dichter."""
from pathlib import Path

from spikeinterface.core.old_api_utils import OldToNewRecording

import spikeextractors as se
from pynwb.ecephys import ElectricalSeries

from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ....utils import get_schema_from_hdmf_class, FilePathType

try:
    from pyintan.intan import read_rhd, read_rhs

    HAVE_PYINTAN = True
except ImportError:
    HAVE_PYINTAN = False
INSTALL_MESSAGE = "Please install pyintan to use this extractor!"


def extract_electrode_metadata_with_pyintan(file_path):
    if ".rhd" in Path(file_path).suffixes:
        intan_file_metadata = read_rhd(file_path)[1]
    else:
        intan_file_metadata = read_rhs(file_path)[1]

    exclude_chan_types = ["AUX", "ADC", "VDD", "_STIM", "ANALOG"]

    valid_channels = [
        x for x in intan_file_metadata if not any([y in x["native_channel_name"] for y in exclude_chan_types])
    ]

    group_names = [channel["native_channel_name"].split("-")[0] for channel in valid_channels]
    unique_group_names = set(group_names)
    group_electrode_numbers = [channel["native_order"] for channel in valid_channels]
    custom_names = [channel["custom_channel_name"] for channel in valid_channels]

    electrodes_metadata = dict(
        group_names=group_names,
        unique_group_names=unique_group_names,
        group_electrode_numbers=group_electrode_numbers,
        custom_names=custom_names,
    )

    return electrodes_metadata


class IntanRecordingInterface(BaseRecordingExtractorInterface):
    """Primary data interface class for converting a IntanRecordingExtractor."""

    RX = se.IntanRecordingExtractor

    def __init__(self, file_path: FilePathType, verbose: bool = True):
        assert HAVE_PYINTAN, INSTALL_MESSAGE
        super().__init__(file_path=file_path, verbose=verbose)

        electrodes_metadata = extract_electrode_metadata_with_pyintan(file_path)
        self.recording_extractor = OldToNewRecording(oldapi_recording_extractor=self.recording_extractor)

        group_names = electrodes_metadata["group_names"]
        group_electrode_numbers = electrodes_metadata["group_electrode_numbers"]
        unique_group_names = electrodes_metadata["unique_group_names"]
        custom_names = electrodes_metadata["custom_names"]

        channel_ids = self.recording_extractor.get_channel_ids()
        self.recording_extractor.set_property(key="group_name", ids=channel_ids, values=group_names)
        if len(unique_group_names) > 1:
            self.recording_extractor.set_property(
                key="group_electrode_number", ids=channel_ids, values=group_electrode_numbers
            )

        if any(custom_names):
            self.recording_extractor.set_property(key="custom_channel_name", ids=channel_ids, values=custom_names)

    def get_metadata_schema(self):
        metadata_schema = super().get_metadata_schema()
        metadata_schema["properties"]["Ecephys"]["properties"].update(
            ElectricalSeries_raw=get_schema_from_hdmf_class(ElectricalSeries)
        )
        return metadata_schema

    def get_metadata(self):
        unique_group_name = set(self.recording_extractor.get_property("group_name"))
        device = dict(
            name="Intan",
            description="Intan recording",
            manufacturer="Intan",
        )

        device_list = [device]

        ecephys_metadata = dict(
            Ecephys=dict(
                Device=device_list,
                ElectrodeGroup=[
                    dict(
                        name=group_name,
                        description=f"Group {group_name} electrodes.",
                        device="Intan",
                        location="",
                    )
                    for group_name in unique_group_name
                ],
                Electrodes=[
                    dict(name="group_name", description="The name of the ElectrodeGroup this electrode is a part of.")
                ],
                ElectricalSeries_raw=dict(name="ElectricalSeries_raw", description="Raw acquisition traces."),
            )
        )

        recording_extractor_properties = self.recording_extractor.get_property_keys()

        if "group_electrode_number" in recording_extractor_properties:
            ecephys_metadata["Ecephys"]["Electrodes"].append(
                dict(name="group_electrode_number", description="0-indexed channel within a group.")
            )
        if "custom_channel_name" in recording_extractor_properties:
            ecephys_metadata["Ecephys"]["Electrodes"].append(
                dict(name="custom_channel_name", description="Custom channel name assigned in Intan.")
            )

        return ecephys_metadata
