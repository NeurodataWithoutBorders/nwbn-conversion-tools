from typing import Union, Optional, Tuple
import distutils.version
import uuid
from datetime import datetime
from pathlib import Path
import warnings

import neo.io.baseio
import pynwb

from .conversion_tools import add_devices

PathType = Union[str, Path, None]


response_classes = {
    "voltage_clamp": pynwb.icephys.VoltageClampSeries,
    "current_clamp": pynwb.icephys.CurrentClampSeries,
    "izero": pynwb.icephys.IZeroClampSeries,
}

stim_classes = {
    "voltage_clamp": pynwb.icephys.VoltageClampStimulusSeries,
    "current_clamp": pynwb.icephys.CurrentClampStimulusSeries,
}


# TODO - get electrodes metadata
def get_electrodes_metadata(neo_reader, electrodes_ids: list, block: int = 0) -> list:
    """
    Get electrodes metadata from Neo reader. The typical information we look for is the information
    accepted by pynwb.icephys.IntracellularElectrode:
    - name – the name of this electrode
    - device – the device that was used to record from this electrode
    - description – Recording description, description of electrode (e.g., whole-cell, sharp, etc) COMMENT: Free-form text (can be from Methods)
    - slice – Information about slice used for recording.
    - seal – Information about seal used for recording.
    - location – Area, layer, comments on estimation, stereotaxis coordinates (if in vivo, etc).
    - resistance – Electrode resistance COMMENT: unit: Ohm.
    - filtering – Electrode specific filtering.
    - initial_access_resistance – Initial access resistance.

    Args:
        neo_reader ([type]): Neo reader
        electrodes_ids (list): List of electrodes ids.
        block (int, optional): Block id. Defaults to 0.

    Returns:
        list: List of dictionaries containing electrodes metadata
    """
    return []


# Get number of electrodes
def get_number_of_electrodes(neo_reader) -> int:
    """
    Get number of electrodes from Neo reader

    Args:
        neo_reader ([type]): Neo reader

    Returns:
        int: number of electrodes
    """
    return len(neo_reader.header["signal_channels"])


# Get number of segments
def get_number_of_segments(neo_reader, block: int = 0) -> int:
    """
    Get number of segments from Neo reader

    Args:
        neo_reader ([type]): Neo reader
        block (int, optional): Block id. Defaults to 0.

    Returns:
        int: number of electrodes
    """
    return neo_reader.header["nb_segment"][block]


# Get command traces (e.g. voltage clamp command traces)
def get_command_traces(neo_reader, block: int = 0, segment: int = 0, cmd_channel: int = 0) -> Tuple[list, str, str]:
    """
    Get command traces (e.g. voltage clamp command traces).

    Args:
        neo_reader ([type]): [description]
        block (int, optional): [description]. Defaults to 0.
        segment (int, optional): [description]. Defaults to 0.
        cmd_channel (int, optional): ABF command channel (0 to 7). Defaults to 0.

    Returns:
        list: [description]
    """
    try:
        traces, titles, units = neo_reader.read_raw_protocol()
        return traces[segment][cmd_channel], titles[segment][cmd_channel], units[segment][cmd_channel]
    except Exception as e:
        msg = ".\n\n WARNING - get_command_traces() only works for AxonIO interface."
        e.args = (str(e) + msg,)
        return e


# Get gain (to Volt or Ampere) from unit in string format
def get_gain_from_unit(unit: str) -> float:
    """
    Get gain (to Volt or Ampere) from unit in string format.

    Args:
        unit (str): Unit as string. E.g. pA, mV, uV, etc...

    Returns:
        float: gain to Ampere or Volt
    """
    if unit in ["pA", "pV"]:
        gain = 10 ** -12
    elif unit in ["nA", "nV"]:
        gain = 10 ** -9
    elif unit in ["uA", "uV"]:
        gain = 10 ** -6
    elif unit in ["mA", "mV"]:
        gain = 10 ** -3
    elif unit in ["A", "V"]:
        gain = 10 ** 0
    else:
        gain = 10 ** 0
        warnings.warn("No valid units found for traces in the current file. Gain is set to 1, but this might be wrong.")
    return float(gain)


# Get basic NWB metadata for Icephys
def get_nwb_metadata(neo_reader, metadata: dict = None):
    """
    Return default metadata for all recording fields.

    Parameters
    ----------
    neo_reader: Neo reader object
    metadata: dict
        metadata info for constructing the nwb file (optional).
    """
    metadata = dict(
        NWBFile=dict(
            session_description="Auto-generated by NwbRecordingExtractor without description.",
            identifier=str(uuid.uuid4()),
            session_start_time=datetime(1970, 1, 1),
        ),
        Icephys=dict(Device=[dict(name="Device", description="no description")]),
    )
    return metadata


# Add Icephys electrode
def add_icephys_electrode(neo_reader, nwbfile=None, metadata: dict = None):
    """
    Adds icephys electrodes to nwbfile object.
    Will always ensure nwbfile has at least one icephys electrode.
    Will auto-generate a linked device if the specified name does not exist in the nwbfile.

    Args:
        neo_reader ([type]): [description]
        nwbfile: NWBFile
            nwb file to which the icephys electrode is to be added
        metadata: dict
            metadata info for constructing the nwb file (optional).
            Should be of the format
                metadata['Icephys']['Electrodes'] = [
                    {
                        'name': my_name,
                        'description': my_description,
                        'device_name': my_device_name
                    },
                    ...
                ]
    """
    if nwbfile is not None:
        assert isinstance(nwbfile, pynwb.NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

    if len(nwbfile.devices) == 0:
        warnings.warn("When adding Icephys Electrode, no Devices were found on nwbfile. Creating a Device now...")
        add_devices(nwbfile=nwbfile, metadata=metadata)

    if metadata is None:
        metadata = dict()

    if "Icephys" not in metadata:
        metadata["Icephys"] = dict()

    defaults = [
        dict(
            name=f"icephys_electrode_{elec_id}",
            description="no description",
            device_name=[i.name for i in nwbfile.devices.values()][0],
        )
        for elec_id in range(get_number_of_electrodes(neo_reader))
    ]

    if "Electrodes" not in metadata["Icephys"] or len(metadata["Icephys"]["Electrodes"]) == 0:
        metadata["Icephys"]["Electrodes"] = defaults

    assert all(
        [isinstance(x, dict) for x in metadata["Icephys"]["Electrodes"]]
    ), "Expected metadata['Icephys']['Electrodes'] to be a list of dictionaries!"

    # Create Icephys electrode from metadata
    for elec in metadata["Icephys"]["Electrodes"]:
        if elec.get("name", defaults[0]["name"]) not in nwbfile.icephys_electrodes:
            device_name = elec.get("device_name", defaults[0]["device_name"])
            if device_name not in nwbfile.devices:
                new_device_metadata = dict(Ecephys=dict(Device=[dict(name=device_name)]))
                add_devices(nwbfile, metadata=new_device_metadata)
                warnings.warn(
                    f"Device '{device_name}' not detected in "
                    "attempted link to icephys electrode! Automatically generating."
                )
            electrode_kwargs = dict(
                name=elec.get("name", defaults[0]["name"]),
                description=elec.get("description", defaults[0]["description"]),
                device=nwbfile.devices[device_name],
            )
            nwbfile.create_icephys_electrode(**electrode_kwargs)


# Add Icephys recordings: stimulus/response pair
def add_icephys_recordings(
    neo_reader,
    nwbfile=None,
    metadata: dict = None,
    icephys_experiment_type: Optional[str] = None,
    stimulus_type="unknown",
):
    """
    Adds icephys recordings (stimulus/response pairs) to nwbfile object.

    Args:
        neo_reader ([type]): [description]
        nwbfile ([type], optional): [description]. Defaults to None.
        metadata (dict, optional): [description]. Defaults to None.
        icephys_experiment_type (str, optional):
            Type of Icephys experiment. Allowed types are: 'voltage_clamp', 'current_clamp' and 'izero'.
            If no value is passed, 'voltage_clamp' is used as default.
    """

    n_segments = get_number_of_segments(neo_reader, block=0)
    n_electrodes = get_number_of_electrodes(neo_reader)
    protocol = neo_reader.read_raw_protocol()

    if icephys_experiment_type is None:
        icephys_experiment_type = "voltage_clamp"

    n_commands = len(protocol[0])
    if n_commands == 0:
        icephys_experiment_type = "izero"
        warnings.warn("No command data found by neo reader. Saving experiment as 'i_zero'...")
    else:
        assert (
            n_commands == n_segments
        ), f"File contains inconsistent number of segments ({n_segments}) and commands ({n_commands})"

    assert icephys_experiment_type in [
        "voltage_clamp",
        "current_clamp",
        "izero",
    ], f"'icephys_experiment_type' should be 'voltage_clamp', 'current_clamp' or 'izero', but received value {icephys_experiment_type}"

    # TODO - check and auto-create devices and electrodes, in case those items don't existe yet on nwbfile

    # Check if nwb object already has sequential recordings
    if getattr(nwbfile, "icephys_sequential_recordings", None):
        offset_sequences = nwbfile.icephys_sequential_recordings.id.data[-1]
    else:
        offset_sequences = 0

    # Loop through segments - sequential icephys recordings
    simultaneous_recordings = list()
    if getattr(nwbfile, "icephys_simultaneous_recordings", None):
        simultaneous_recordings_offset = len(nwbfile.icephys_simultaneous_recordings)
    else:
        simultaneous_recordings_offset = 0

    for si in range(n_segments):
        si_o = offset_sequences + si + 1
        # Loop through electrodes - parallel icephys recordings
        recordings = list()
        for ei, electrode in enumerate(
            list(nwbfile.icephys_electrodes.values())[: len(neo_reader.header["signal_channels"]["units"])]
        ):
            sampling_rate = neo_reader.get_signal_sampling_rate()
            starting_time = neo_reader.get_signal_t_start(block_index=0, seg_index=si)
            response_unit = neo_reader.header["signal_channels"]["units"][ei]
            response_gain = get_gain_from_unit(unit=response_unit)
            response_name = f"{icephys_experiment_type}_response_{si_o + simultaneous_recordings_offset}_ch_{ei}"

            response = response_classes[icephys_experiment_type](
                name=response_name,
                electrode=electrode,
                data=neo_reader.get_analogsignal_chunk(block_index=0, seg_index=si, channel_indexes=ei),
                starting_time=starting_time,
                rate=sampling_rate,
                conversion=response_gain,
                gain=1.0,
            )
            if icephys_experiment_type != "izero":
                stim_unit = protocol[2][ei]
                stim_gain = get_gain_from_unit(unit=stim_unit)
                stimulus = stim_classes[icephys_experiment_type](
                    name=f"stimulus-{si_o + simultaneous_recordings_offset}-ch-{ei}",
                    electrode=electrode,
                    data=protocol[0][si][ei],
                    rate=sampling_rate,
                    starting_time=starting_time,
                    conversion=stim_gain,
                    gain=1.0,
                )
                icephys_recording = nwbfile.add_intracellular_recording(
                    electrode=electrode, response=response, stimulus=stimulus
                )
            else:
                icephys_recording = nwbfile.add_intracellular_recording(electrode=electrode, response=response)

            # Add channel sweep to list
            recordings.append(icephys_recording)

        # Add a list of sweeps to the simultaneous recordings table
        sim_rec = nwbfile.add_icephys_simultaneous_recording(recordings=recordings)
        simultaneous_recordings.append(sim_rec)

    # Add a list of simultaneous recordings table indices as a sequential recording
    seq_rec = nwbfile.add_icephys_sequential_recording(
        simultaneous_recordings=simultaneous_recordings, stimulus_type=stimulus_type
    )

    # Add a list of sequential recordings table indices as a repetition
    run_index = nwbfile.add_icephys_repetition(
        sequential_recordings=[
            seq_rec,
        ]
    )

    # Add a list of repetition table indices as a experimental condition
    nwbfile.add_icephys_experimental_condition(
        repetitions=[
            run_index,
        ]
    )


def add_all_to_nwbfile(
    neo_reader,
    nwbfile=None,
    use_times: bool = False,
    metadata: dict = None,
    write_as: str = "raw",
    es_key: str = None,
    write_scaled: bool = False,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = None,
    iterator_type: Optional[str] = None,
    iterator_opts: Optional[dict] = None,
    icephys_experiment_type: Optional[str] = "voltage_clamp",
):
    """
    Auxiliary static method for nwbextractor.

    Adds all recording related information from recording object and metadata to the nwbfile object.

    Parameters
    ----------
    neo_reader: Neo reader object
    nwbfile: NWBFile
        nwb file to which the recording information is to be added
    use_times: bool
        If True, the times are saved to the nwb file using recording.frame_to_time(). If False (defualut),
        the sampling rate is used.
    metadata: dict
        metadata info for constructing the nwb file (optional).
        Check the auxiliary function docstrings for more information
        about metadata format.
    write_as: str (optional, defaults to 'raw')
        How to save the traces data in the nwb file. Options:
        - 'raw' will save it in acquisition
        - 'processed' will save it as FilteredEphys, in a processing module
        - 'lfp' will save it as LFP, in a processing module
    es_key: str (optional)
        Key in metadata dictionary containing metadata info for the specific electrical series
    write_scaled: bool (optional, defaults to True)
        If True, writes the scaled traces (return_scaled=True)
    compression: str (optional, defaults to "gzip")
        Type of compression to use. Valid types are "gzip" and "lzf".
        Set to None to disable all compression.
    compression_opts: int (optional, defaults to 4)
        Only applies to compression="gzip". Controls the level of the GZIP.
    iterator_type: str (optional, defaults to 'v2')
        The type of DataChunkIterator to use.
        'v1' is the original DataChunkIterator of the hdmf data_utils.
        'v2' is the locally developed RecordingExtractorDataChunkIterator, which offers full control over chunking.
    iterator_opts: dict (optional)
        Dictionary of options for the RecordingExtractorDataChunkIterator (iterator_type='v2')
        or DataChunkIterator (iterator_tpye='v1').
        Valid options are
            buffer_gb : float (optional, defaults to 1 GB, available for both 'v2' and 'v1')
                Recommended to be as much free RAM as available). Automatically calculates suitable buffer shape.
            chunk_mb : float (optional, defaults to 1 MB, only available for 'v2')
                Should be below 1 MB. Automatically calculates suitable chunk shape.
        If manual specification of buffer_shape and chunk_shape are desired, these may be specified as well.
    icephys_experiment_type: str (optional)
        Type of Icephys experiment. Allowed types are: 'voltage_clamp', 'current_clamp' and 'izero'.
        If no value is passed, 'voltage_clamp' is used as default.
    """
    if nwbfile is not None:
        assert isinstance(nwbfile, pynwb.NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

    add_devices(
        nwbfile=nwbfile, 
        data_type="Icephys", 
        metadata=metadata
    )
    
    add_icephys_electrode(
        neo_reader=neo_reader,
        nwbfile=nwbfile,
        metadata=metadata,
    )

    add_icephys_recordings(
        neo_reader=neo_reader,
        nwbfile=nwbfile,
        metadata=metadata,
        icephys_experiment_type=icephys_experiment_type,
    )


def write_neo_to_nwb(
    neo_reader: neo.io.baseio.BaseIO,
    save_path: PathType = None,
    overwrite: bool = False,
    nwbfile=None,
    use_times: bool = False,
    metadata: dict = None,
    write_as: str = "raw",
    es_key: str = None,
    write_scaled: bool = False,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = None,
    iterator_type: Optional[str] = None,
    iterator_opts: Optional[dict] = None,
    icephys_experiment_type: Optional[str] = None,
):
    """
    Primary method for writing a Neo reader object to an NWBFile.

    Parameters
    ----------
    neo_reader: Neo reader
    save_path: PathType
        Required if an nwbfile is not passed. Must be the path to the nwbfile
        being appended, otherwise one is created and written.
    overwrite: bool
        If using save_path, whether or not to overwrite the NWBFile if it already exists.
    nwbfile: NWBFile
        Required if a save_path is not specified. If passed, this function
        will fill the relevant fields within the nwbfile.
    use_times: bool
        If True, the times are saved to the nwb file. If False (default), the sampling rate is used.
    metadata: dict
        metadata info for constructing the nwb file (optional). Should be of the format
            metadata['Ecephys'] = {}
        with keys of the forms
            metadata['Ecephys']['Device'] = [
                {
                    'name': my_name,
                    'description': my_description
                },
                ...
            ]
            metadata['Ecephys']['ElectrodeGroup'] = [
                {
                    'name': my_name,
                    'description': my_description,
                    'location': electrode_location,
                    'device': my_device_name
                },
                ...
            ]
            metadata['Ecephys']['Electrodes'] = [
                {
                    'name': my_name,
                    'description': my_description
                },
                ...
            ]
            metadata['Ecephys']['ElectricalSeries'] = {
                'name': my_name,
                'description': my_description
            }

        Note that data intended to be added to the electrodes table of the NWBFile should be set as channel
        properties in the RecordingExtractor object.
    write_as: str (optional, defaults to 'raw')
        How to save the traces data in the nwb file. Options:
        - 'raw' will save it in acquisition
        - 'processed' will save it as FilteredEphys, in a processing module
        - 'lfp' will save it as LFP, in a processing module
    es_key: str (optional)
        Key in metadata dictionary containing metadata info for the specific electrical series
    write_scaled: bool (optional, defaults to True)
        If True, writes the scaled traces (return_scaled=True)
    compression: str (optional, defaults to "gzip")
        Type of compression to use. Valid types are "gzip" and "lzf".
        Set to None to disable all compression.
    compression_opts: int (optional, defaults to 4)
        Only applies to compression="gzip". Controls the level of the GZIP.
    iterator_type: str (optional, defaults to 'v2')
        The type of DataChunkIterator to use.
        'v1' is the original DataChunkIterator of the hdmf data_utils.
        'v2' is the locally developed RecordingExtractorDataChunkIterator, which offers full control over chunking.
    iterator_opts: dict (optional)
        Dictionary of options for the RecordingExtractorDataChunkIterator (iterator_type='v2').
        Valid options are
            buffer_gb : float (optional, defaults to 1 GB)
                Recommended to be as much free RAM as available). Automatically calculates suitable buffer shape.
            chunk_mb : float (optional, defaults to 1 MB)
                Should be below 1 MB. Automatically calculates suitable chunk shape.
        If manual specification of buffer_shape and chunk_shape are desired, these may be specified as well.
    icephys_experiment_type: str (optional)
        Type of Icephys experiment. Allowed types are: 'voltage_clamp', 'current_clamp' and 'izero'.
        If no value is passed, 'voltage_clamp' is used as default.
    """
    if nwbfile is not None:
        assert isinstance(nwbfile, pynwb.NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

    assert (
        distutils.version.LooseVersion(pynwb.__version__) >= "1.3.3"
    ), "'write_neo_to_nwb' not supported for version < 1.3.3. Run pip install --upgrade pynwb"

    assert save_path is None or nwbfile is None, "Either pass a save_path location, or nwbfile object, but not both!"

    if metadata is None:
        metadata = get_nwb_metadata(neo_reader=neo_reader)

    if nwbfile is None:
        if Path(save_path).is_file() and not overwrite:
            read_mode = "r+"
        else:
            read_mode = "w"

        with pynwb.NWBHDF5IO(str(save_path), mode=read_mode) as io:
            if read_mode == "r+":
                nwbfile = io.read()
            else:
                nwbfile_kwargs = dict(
                    session_description="Auto-generated by NwbRecordingExtractor without description.",
                    identifier=str(uuid.uuid4()),
                    session_start_time=datetime(1970, 1, 1),
                )
                if metadata is not None and "NWBFile" in metadata:
                    nwbfile_kwargs.update(metadata["NWBFile"])
                nwbfile = pynwb.NWBFile(**nwbfile_kwargs)

            add_all_to_nwbfile(
                neo_reader=neo_reader,
                nwbfile=nwbfile,
                metadata=metadata,
                use_times=use_times,
                write_as=write_as,
                es_key=es_key,
                write_scaled=write_scaled,
                compression=compression,
                compression_opts=compression_opts,
                iterator_type=iterator_type,
                iterator_opts=iterator_opts,
                icephys_experiment_type=icephys_experiment_type,
            )
            io.write(nwbfile)
    else:
        add_all_to_nwbfile(
            neo_reader=neo_reader,
            nwbfile=nwbfile,
            use_times=use_times,
            metadata=metadata,
            write_as=write_as,
            es_key=es_key,
            write_scaled=write_scaled,
            compression=compression,
            compression_opts=compression_opts,
            iterator_type=iterator_type,
            iterator_opts=iterator_opts,
            icephys_experiment_type=icephys_experiment_type,
        )
