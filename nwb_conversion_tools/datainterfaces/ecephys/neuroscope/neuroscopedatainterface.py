"""Authors: Cody Baker and Ben Dichter."""
from pathlib import Path

import numpy as np
import spikeextractors as se

from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ..baselfpextractorinterface import BaseLFPExtractorInterface
from ..basesortingextractorinterface import BaseSortingExtractorInterface

try:
    from lxml import etree as et
    HAVE_LXML = True
except ImportError:
    HAVE_LXML = False
INSTALL_MESSAGE = "Please install lxml to use this extractor!"


def get_xml_file_path(data_file_path: str):
    """
    Infer the xml_file_path from the data_file_path (.dat or .eeg).

    Assumes the two are in the same folder and follow the session_id naming convention.
    """
    session_path = Path(data_file_path).parent
    session_id = session_path.stem
    return str((session_path / f"{session_id}.xml").absolute())


def get_xml(xml_file_path: str):
    """Auxiliary function for retrieving root of xml."""
    root = et.parse(xml_file_path).getroot()

    return root


def get_shank_channels(xml_file_path: str, sort: bool = False):
    """
    Auxiliary function for retrieving the list of structured shank-only channels.

    Attempts to retrieve these first from the spikeDetection sub-field in the event that spike sorting was performed on
    the raw data. In the event that spike sorting was not performed, it then retrieves only the anatomicalDescription.
    """
    root = get_xml(xml_file_path)
    try:
        shank_channels = [
            [int(channel.text) for channel in group.find('channels')]
            for group in root.find('spikeDetection').find('channelGroups').findall('group')
        ]
    except (TypeError, AttributeError):
        shank_channels = [
            [int(channel.text) for channel in group.findall('channel')]
            for group in root.find('anatomicalDescription').find('channelGroups').findall('group')
        ]

    if sort:
        shank_channels = sorted(np.concatenate(shank_channels))

    return shank_channels


class NeuroscopeRecordingInterface(BaseRecordingExtractorInterface):
    """Primary data interface class for converting a NeuroscopeRecordingExtractor."""

    RX = se.NeuroscopeRecordingExtractor

    @staticmethod
    def get_ecephys_metadata(xml_file_path: str, recording):
        """Auto-populates ecephys metadata from the xml_file_path inferred."""
        session_path = Path(xml_file_path).parent
        session_id = session_path.stem
        shank_channels = get_shank_channels(xml_file_path)

        elec_group_names = [f"Shank{n + 1}" for n, channels in enumerate(shank_channels) for _ in channels]
        shank_electrode_numbers = [x for channels in shank_channels for x, _ in enumerate(channels)]
        for channel_id, channel_group, shank_electrode_number in zip(
                self.recording_extractor.get_channel_ids(),
                elec_group_names,
                shank_electrode_numbers
        ):
            self.recording_extractor.set_channel_property(
                channel_id=channel_id,
                property_name="group_name",
                value=f"Group{channel_group}"
            )
            self.recording_extractor.set_channel_property(
                channel_id=channel_id,
                property_name="shank_electrode_number",
                value=shank_electrode_number
            )

        ecephys_metadata = dict(
            Ecephys=dict(
                Device=[
                    dict(
                        description=session_id + '.xml'
                    )
                ],
                ElectrodeGroup=[
                    dict(
                        name=f'Shank{n + 1}',
                        description=f"Shank{n + 1} electrodes"
                    )
                    for n, _ in enumerate(shank_channels)
                ],
                Electrodes=[
                    dict(
                        name='shank_electrode_number',
                        description="0-indexed channel within a shank."
                    ),
                    dict(
                        name='group_name',
                        description="The name of the ElectrodeGroup this electrode is a part of."
                    )
                ]
            )
        )

        return ecephys_metadata

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subset_channels = get_shank_channels(
            xml_file_path=get_xml_file_path(data_file_path=self.source_data['file_path']),
            sort=True
        )

    def get_metadata(self):
        """Retrieve Ecephys metadata specific to the Neuroscope format."""
        metadata = NeuroscopeRecordingInterface.get_ecephys_metadata(
            xml_file_path=get_xml_file_path(data_file_path=self.source_data['file_path']),
            recording=self.recording_extractor
        )
        metadata['Ecephys'].update(
            ElectricalSeries=dict(
                name='ElectricalSeries',
                description="Raw acquisition traces."
            )
        )

        return metadata


class NeuroscopeMultiRecordingTimeInterface(BaseRecordingExtractorInterface):
    """Primary data interface class for converting a NeuroscopeMultiRecordingTimeExtractor."""

    RX = se.NeuroscopeMultiRecordingTimeExtractor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subset_channels = get_shank_channels(
            xml_file_path=get_xml_file_path(data_file_path=self.source_data['folder_path']),
            sort=True
        )
        
    def get_metadata(self):
        """Retrieve Ecephys metadata specific to the Neuroscope format."""
        metadata = NeuroscopeRecordingInterface.get_ecephys_metadata(
            xml_file_path=get_xml_file_path(data_file_path=self.source_data['folder_path'])
        )
        metadata['Ecephys'].update(
            ElectricalSeries=dict(
                name='ElectricalSeries',
                description="Raw acquisition traces."
            )
        )

        return metadata


class NeuroscopeLFPInterface(BaseLFPExtractorInterface):
    """Primary data interface class for converting Neuroscope LFP data."""

    RX = se.NeuroscopeRecordingExtractor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subset_channels = get_shank_channels(
            xml_file_path=get_xml_file_path(data_file_path=self.source_data['file_path']),
            sort=True
        )

    def get_metadata(self):
        """Retrieve Ecephys metadata specific to the Neuroscope format."""
        metadata = NeuroscopeRecordingInterface.get_ecephys_metadata(
            xml_file_path=get_xml_file_path(data_file_path=self.source_data['file_path']),
            recording=self.recording_extractor
        )
        metadata['Ecephys'].update(
            ElectricalSeries_lfp=dict(
                name="LFP",
                description="Local field potential signal."
            )
        )

        return metadata


class NeuroscopeSortingInterface(BaseSortingExtractorInterface):
    """Primary data interface class for converting a NeuroscopeSortingExtractor."""

    SX = se.NeuroscopeMultiSortingExtractor

    def get_metadata(self):
        """Auto-populates spiking unit metadata."""
        session_path = Path(self.source_data['folder_path'])
        session_id = session_path.stem
        metadata = NeuroscopeRecordingInterface.get_ecephys_metadata(
            xml_file_path=str(session_path / f"{session_id}.xml"),
            recording=NumpyRecordingExtractor(
                timeseries=np.array([]),
                sampling_frequency=self.sorting_extractor.get_sampling_frequency
            )
            # TODO, with new recording channel property format for heavy metadata, need to call add_electrodes
            # on the data contained in this dummy RecordingExtractor
        )
        metadata.update(UnitProperties=[])
        return metadata
