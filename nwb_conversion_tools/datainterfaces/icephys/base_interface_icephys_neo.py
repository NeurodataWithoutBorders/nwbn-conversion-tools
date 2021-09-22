from abc import ABC
from typing import Union, Optional, Tuple
from pathlib import Path
import numpy as np

import spikeextractors as se
from pynwb import NWBFile
from pynwb.device import Device
from pynwb.icephys import IntracellularElectrode

from ...basedatainterface import BaseDataInterface
from ...utils.json_schema import (
    get_schema_from_hdmf_class,
    get_schema_from_method_signature,
    fill_defaults,
    get_base_schema,
)
from ...utils.spike_interface import write_recording
from ...utils.neo import (
    get_command_traces,
    get_number_of_electrodes,
    get_electrodes_metadata,
    get_number_of_segments,
    write_neo_to_nwb,
)

OptionalPathType = Optional[Union[str, Path]]


class BaseIcephysNeoInterface(BaseDataInterface, ABC):
    """Primary class for all NeoInterfaces."""

    neo_class = None

    @classmethod
    def get_source_schema(cls):
        """Compile input schema for the Neo class"""
        source_schema = get_schema_from_method_signature(class_method=cls.__init__, exclude=[])
        return source_schema

    def __init__(self, **source_data):
        super().__init__(**source_data)
        self.source_data = source_data

        self.reader = self.neo_class(**source_data)
        self.subset_channels = None
        self.n_segments = get_number_of_segments(neo_reader=self.reader, block=0)
        self.n_channels = get_number_of_electrodes(neo_reader=self.reader)

    def get_metadata_schema(self):
        """Compile metadata schema for the Neo interface"""
        metadata_schema = super().get_metadata_schema()

        # Initiate Ecephys metadata
        metadata_schema["properties"]["Icephys"] = get_base_schema(tag="Icephys")
        metadata_schema["properties"]["Icephys"]["required"] = ["Device", "IntracellularElectrode"]
        metadata_schema["properties"]["Icephys"]["properties"] = dict(
            Device=dict(type="array", minItems=1, items={"$ref": "#/properties/Icephys/properties/definitions/Device"}),
            IntracellularElectrode=dict(
                type="array",
                minItems=1,
                items={"$ref": "#/properties/Icephys/properties/definitions/IntracellularElectrode"},
            ),
        )
        # Schema definition for arrays
        metadata_schema["properties"]["Icephys"]["properties"]["definitions"] = dict(
            Device=get_schema_from_hdmf_class(Device),
            IntracellularElectrode=get_schema_from_hdmf_class(IntracellularElectrode),
        )
        return metadata_schema

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["Icephys"] = dict(
            Device=[dict(name="Device_icephys", description="no description")],
            IntracellularElectrode=[
                dict(name=f"electrode-{i}", description="no description", device="Device_icephys")
                for i in range(get_number_of_electrodes(self.reader))
            ],
        )
        return metadata

    def run_conversion(
        self,
        nwbfile: NWBFile,
        metadata: dict = None,
        stub_test: bool = False,
        use_times: bool = False,
        save_path: OptionalPathType = None,
        overwrite: bool = False,
        write_as: str = "raw",
        es_key: str = None,
        icephys_experiment_type: Optional[str] = None,
    ):
        """
        Primary function for converting raw (unprocessed) RecordingExtractor data to the NWB standard.

        Parameters
        ----------
        nwbfile: NWBFile
            nwb file to which the recording information is to be added
        metadata: dict
            metadata info for constructing the nwb file (optional).
            Should be of the format
                metadata['Ecephys']['ElectricalSeries'] = dict(name=my_name, description=my_description)
        use_times: bool
            If True, the times are saved to the nwb file using recording.frame_to_time(). If False (default),
            the sampling rate is used.
        save_path: PathType
            Required if an nwbfile is not passed. Must be the path to the nwbfile
            being appended, otherwise one is created and written.
        overwrite: bool
            If using save_path, whether or not to overwrite the NWBFile if it already exists.
        stub_test: bool, optional (default False)
            If True, will truncate the data to run the conversion faster and take up less memory.
        buffer_mb: int (optional, defaults to 500MB)
            Maximum amount of memory (in MB) to use per iteration of the internal DataChunkIterator.
            Requires trace data in the RecordingExtractor to be a memmap object.
        write_as: str (optional, defaults to 'raw')
            Options: 'raw', 'lfp' or 'processed'
        es_key: str (optional)
            Key in metadata dictionary containing metadata info for the specific electrical series
        icephys_experiment_type: str (optional)
            Type of Icephys experiment. Allowed types are: 'voltage_clamp', 'current_clamp' and 'izero'.
            If no value is passed, 'voltage_clamp' is used as default.
        """
        # TODO - stub test
        # if stub_test or self.subset_channels is not None:
        #     recording = self.subset_recording(stub_test=stub_test)
        # else:
        #     recording = self.recording_extractor

        write_neo_to_nwb(
            neo_reader=self.reader,
            nwbfile=nwbfile,
            metadata=metadata,
            use_times=use_times,
            write_as=write_as,
            es_key=es_key,
            save_path=save_path,
            overwrite=overwrite,
            icephys_experiment_type=icephys_experiment_type,
        )
