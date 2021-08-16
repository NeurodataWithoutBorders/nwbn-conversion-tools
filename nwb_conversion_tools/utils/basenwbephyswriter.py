import pynwb
import numpy as np
import uuid
from datetime import datetime
from copy import deepcopy
from numbers import Real
import warnings
import distutils.version
from .json_schema import dict_deep_update
from collections import Iterable, defaultdict
from .common_writer_tools import default_export_ops, ArrayType, _default_sorting_property_descriptions
from abc import ABC, abstractmethod
import psutil
from warnings import warn
from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5.h5_utils import H5DataIO


class BaseNwbEphysWriter(ABC):
    def __init__(self, object_to_write, nwbfile=None, metadata=None, **kwargs):
        self.object_to_write = object_to_write
        assert nwbfile is not None and isinstance(nwbfile, pynwb.NWBFile), "Instantiate an NWBFile and pass as argument"
        self.metadata = metadata if metadata is not None else dict()
        self.nwbfile = nwbfile
        self._conversion_ops = dict(**default_export_ops(), **kwargs)

    @abstractmethod
    def _get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        pass

    @abstractmethod
    def _get_channel_ids(self):
        pass

    @abstractmethod
    def _get_channel_property_names(self, chan_id):
        pass

    @abstractmethod
    def _get_channel_property_values(self, prop, chan_id):
        pass

    @abstractmethod
    def _get_unit_ids(self):
        pass

    @abstractmethod
    def _get_unit_sampling_frequency(self):
        pass

    @abstractmethod
    def _get_unit_property_names(self, unit_id):
        pass

    @abstractmethod
    def _get_unit_property_values(self, prop, unit_id):
        pass

    @abstractmethod
    def _get_unit_feature_names(self, unit_id):
        pass

    @abstractmethod
    def _get_unit_feature_values(self, prop, unit_id):
        pass

    @abstractmethod
    def _get_unit_spike_train_ids(self, unit_id):
        pass

    @abstractmethod
    def _get_unit_spike_train_times(self, unit_id):
        pass

    @abstractmethod
    def _get_sampling_frequency(self):
        pass

    @abstractmethod
    def _get_num_frames(self):
        pass

    @abstractmethod
    def _get_times(self):
        pass

    @abstractmethod
    def add_to_nwb(self):
        pass

    def add_recording(self):
        assert (
            distutils.version.LooseVersion(pynwb.__version__) >= "1.3.3"
        ), "'write_recording' not supported for version < 1.3.3. Run pip install --upgrade pynwb"

        self.add_devices()
        self.add_electrode_groups()
        self.add_electrodes()
        if self._conversion_ops["write_electrical_series"]:
            self.add_electrical_series()
            self.add_epochs()

    def add_sorting(self):
        self.add_units()

    @abstractmethod
    def add_epochs(self):
        pass

    @abstractmethod
    def add_waveforms(self):
        pass

    def add_devices(self):
        """
        Auxiliary static method for nwbextractor.

        Adds device information to nwbfile object.
        Will always ensure nwbfile has at least one device, but multiple
        devices within the metadata list will also be created.

        Missing keys in an element of metadata['Ecephys']['Device'] will be auto-populated with defaults.
        """
        if self.nwbfile is not None:
            assert isinstance(self.nwbfile, pynwb.NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

        # Default Device metadata
        defaults = dict(name="Device", description="Ecephys probe. Automatically generated.")

        if "Ecephys" not in self.metadata:
            self.metadata["Ecephys"] = dict()

        if "Device" not in self.metadata["Ecephys"]:
            self.metadata["Ecephys"]["Device"] = [defaults]

        for dev in self.metadata["Ecephys"]["Device"]:
            if dev.get("name", defaults["name"]) not in self.nwbfile.devices:
                self.nwbfile.create_device(**dict(defaults, **dev))

        print(self.nwbfile.devices)

    @abstractmethod
    def add_electrode_groups(self):
        pass

    def _add_electrode_groups(self, channel_groups_unique: Iterable = None):
        """
        Auxiliary method to write electrode groups.

        Adds electrode group information to nwbfile object.
        Will always ensure nwbfile has at least one electrode group.
        Will auto-generate a linked device if the specified name does not exist in the nwbfile.

        Missing keys in an element of metadata['Ecephys']['ElectrodeGroup'] will be auto-populated with defaults.

        Group names set by RecordingExtractor channel properties will also be included with passed metadata,
        but will only use default description and location.
        """
        if len(self.nwbfile.devices) == 0:
            warnings.warn("When adding ElectrodeGroup, no Devices were found on nwbfile. Creating a Device now...")
            self.add_devices()

        if "Ecephys" not in self.metadata:
            self.metadata["Ecephys"] = dict()

        if channel_groups_unique is None:
            channel_groups_unique = np.array([0], dtype="int")

        defaults = [
            dict(
                name=str(group_id),
                description="no description",
                location="unknown",
                device=[i.name for i in self.nwbfile.devices.values()][0],
            )
            for group_id in channel_groups_unique
        ]

        if "ElectrodeGroup" not in self.metadata["Ecephys"]:
            self.metadata["Ecephys"]["ElectrodeGroup"] = defaults

        assert all(
            [isinstance(x, dict) for x in self.metadata["Ecephys"]["ElectrodeGroup"]]
        ), "Expected metadata['Ecephys']['ElectrodeGroup'] to be a list of dictionaries!"

        for grp in self.metadata["Ecephys"]["ElectrodeGroup"]:
            if grp.get("name", defaults[0]["name"]) not in self.nwbfile.electrode_groups:
                device_name = grp.get("device", defaults[0]["device"])
                if device_name not in self.nwbfile.devices:
                    self.metadata["Ecephys"]["Device"].append(dict(name=device_name))
                    self.add_devices()
                    warnings.warn(
                        f"Device '{device_name}' not detected in "
                        "attempted link to electrode group! Automatically generating."
                    )
                electrode_group_kwargs = dict(defaults[0], **grp)
                electrode_group_kwargs.update(device=self.nwbfile.devices[device_name])
                self.nwbfile.create_electrode_group(**electrode_group_kwargs)

    def add_electrodes(self):
        """
        Auxiliary static method for nwbextractor.

        Adds channels from recording object as electrodes to nwbfile object.

        Missing keys in an element of metadata['Ecephys']['ElectrodeGroup'] will be auto-populated with defaults
        whenever possible.

        If 'my_name' is set to one of the required fields for nwbfile
        electrodes (id, x, y, z, imp, loccation, filtering, group_name),
        then the metadata will override their default values.

        Setting 'my_name' to metadata field 'group' is not supported as the linking to
        nwbfile.electrode_groups is handled automatically; please specify the string 'group_name' in this case.

        If no group information is passed via metadata, automatic linking to existing electrode groups,
        possibly including the default, will occur.
        """
        if self.nwbfile.electrodes is not None:
            ids_absent = [id not in self.nwbfile.electrodes.id for id in self._get_channel_ids()]
            if not all(ids_absent):
                warnings.warn("cannot create electrodes for this recording as ids already exist")
                return

        if self.nwbfile is not None:
            assert isinstance(self.nwbfile, pynwb.NWBFile), "'nwbfile' should be of type pynwb.NWBFile"
        if self.nwbfile.electrode_groups is None or len(self.nwbfile.electrode_groups) == 0:
            self.add_electrode_groups()
        # For older versions of pynwb, we need to manually add these columns
        if distutils.version.LooseVersion(pynwb.__version__) < "1.3.0":
            if self.nwbfile.electrodes is None or "rel_x" not in self.nwbfile.electrodes.colnames:
                self.nwbfile.add_electrode_column("rel_x", "x position of electrode in electrode group")
            if self.nwbfile.electrodes is None or "rel_y" not in self.nwbfile.electrodes.colnames:
                self.nwbfile.add_electrode_column("rel_y", "y position of electrode in electrode group")

        defaults = dict(
            x=np.nan,
            y=np.nan,
            z=np.nan,
            # There doesn't seem to be a canonical default for impedence, if missing.
            # The NwbRecordingExtractor follows the -1.0 convention, other scripts sometimes use np.nan
            imp=-1.0,
            location="unknown",
            filtering="none",
            group_name="0",
        )
        if self.metadata is None:
            self.metadata = dict(Ecephys=dict())

        if "Ecephys" not in self.metadata:
            self.metadata["Ecephys"] = dict()

        if "Electrodes" not in self.metadata["Ecephys"]:
            self.metadata["Ecephys"]["Electrodes"] = []

        assert all(
            [
                isinstance(x, dict) and set(x.keys()) == set(["name", "description"])
                for x in self.metadata["Ecephys"]["Electrodes"]
            ]
        ), (
            "Expected metadata['Ecephys']['Electrodes'] to be a list of dictionaries, "
            "containing the keys 'name' and 'description'"
        )
        assert all(
            [x["name"] != "group" for x in self.metadata["Ecephys"]["Electrodes"]]
        ), "Passing metadata field 'group' is deprecated; pass group_name instead!"

        if self.nwbfile.electrodes is None:
            nwb_elec_ids = []
        else:
            nwb_elec_ids = self.nwbfile.electrodes.id.data[:]

        elec_columns = defaultdict(dict)  # dict(name: dict(description='',data=data, index=False))
        elec_columns_append = defaultdict(dict)
        property_names = set()
        for chan_id in self._get_channel_ids():
            for i in self._get_channel_property_names(chan_id):
                property_names.add(i)

        # property 'brain_area' of RX channels corresponds to 'location' of NWB electrodes
        exclude_names = set(["location", "group"] + list(self._conversion_ops["skip_electrode_properties"]))

        channel_property_defaults = {list: [], np.ndarray: np.array(np.nan), str: "", Real: np.nan}
        found_property_types = {prop: Real for prop in property_names}

        for prop in property_names:
            prop_skip = False
            if prop not in exclude_names:
                data = []
                prop_chan_count = 0
                # build data:
                for chan_id in self._get_channel_ids():
                    if prop in self._get_channel_property_names(chan_id):
                        prop_chan_count += 1
                        chan_data = self._get_channel_property_values(prop, chan_id)
                        # find the type and store (only when the first channel with given property is found):
                        if prop_chan_count == 1:
                            proptype = [
                                proptype for proptype in channel_property_defaults if isinstance(chan_data, proptype)
                            ]
                            if len(proptype) > 0:
                                found_property_types[prop] = proptype[0]
                                # cast as float if any number:
                                if found_property_types[prop] == Real:
                                    chan_data = np.float(chan_data)
                                # update data if wrong datatype items filled prior:
                                if len(data) > 0 and not isinstance(data[-1], found_property_types[prop]):
                                    data = [channel_property_defaults[found_property_types[prop]]] * len(data)
                            else:
                                prop_skip = True  # skip storing that property if not of default type
                                break
                        data.append(chan_data)
                    else:
                        data.append(channel_property_defaults[found_property_types[prop]])
                # store data after build:
                if not prop_skip:
                    index = found_property_types[prop] == ArrayType
                    prop_name_new = "location" if prop == "brain_area" else prop
                    found_property_types[prop_name_new] = found_property_types.pop(prop)
                    elec_columns[prop_name_new].update(description=prop_name_new, data=data, index=index)

        for x in self.metadata["Ecephys"]["Electrodes"]:
            elec_columns[x["name"]]["description"] = x["description"]
            if x["name"] not in list(elec_columns):
                raise ValueError(f'"{x["name"]}" not a property of se object')

        # updating default arguments if electrodes table already present:
        default_updated = dict()
        if self.nwbfile.electrodes is not None:
            for colname in self.nwbfile.electrodes.colnames:
                if colname != "group":
                    samp_data = self.nwbfile.electrodes[colname].data[0]
                    default_datatype = [
                        proptype for proptype in channel_property_defaults if isinstance(samp_data, proptype)
                    ][0]
                    default_updated.update({colname: channel_property_defaults[default_datatype]})
        default_updated.update(defaults)

        for name, des_dict in elec_columns.items():
            des_args = dict(des_dict)
            if name not in default_updated:
                if self.nwbfile.electrodes is None:
                    self.nwbfile.add_electrode_column(
                        name=name, description=des_args["description"], index=des_args["index"]
                    )
                else:
                    # build default junk values for data to force add columns later:
                    combine_data = [channel_property_defaults[found_property_types[name]]] * len(
                        self.nwbfile.electrodes.id
                    )
                    des_args["data"] = combine_data + des_args["data"]
                    elec_columns_append[name] = des_args

        for name in elec_columns_append:
            _ = elec_columns.pop(name)

        for j, channel_id in enumerate(self._get_channel_ids()):
            if channel_id not in nwb_elec_ids:
                electrode_kwargs = dict(default_updated)
                electrode_kwargs.update(id=channel_id)

                # self.recording.get_channel_locations defaults to np.nan if there are none
                location = self._get_channel_property_values("location", channel_id)
                if all([not np.isnan(loc) for loc in location]):
                    # property 'location' of RX channels corresponds to rel_x and rel_ y of NWB electrodes
                    electrode_kwargs.update(dict(rel_x=float(location[0]), rel_y=float(location[1])))

                for name, desc in elec_columns.items():
                    if name == "group_name":
                        group_name = str(desc["data"][j])
                        if group_name != "" and group_name not in self.nwbfile.electrode_groups:
                            warnings.warn(
                                f"Electrode group {group_name} for electrode {channel_id} was not "
                                "found in the nwbfile! Automatically adding."
                            )
                            missing_group_metadata = dict(
                                Ecephys=dict(
                                    ElectrodeGroup=[
                                        dict(
                                            name=group_name,
                                        )
                                    ]
                                )
                            )
                            self.add_electrode_groups(missing_group_metadata=missing_group_metadata)
                        electrode_kwargs.update(
                            dict(group=self.nwbfile.electrode_groups[group_name], group_name=group_name)
                        )
                    elif "data" in desc:
                        electrode_kwargs[name] = desc["data"][j]

                if "group_name" not in elec_columns:
                    group_id = self._get_channel_property_values("group", channel_id)[0]
                    electrode_kwargs.update(
                        dict(group=self.nwbfile.electrode_groups[str(group_id)], group_name=str(group_id))
                    )

                self.nwbfile.add_electrode(**electrode_kwargs)
        # add columns for existing electrodes:
        for col_name, cols_args in elec_columns_append.items():
            self.nwbfile.add_electrode_column(col_name, **cols_args)
        assert (
            self.nwbfile.electrodes is not None
        ), "Unable to form electrode table! Check device, electrode group, and electrode metadata."

    def add_electrical_series(self):
        """
        Auxiliary static method for nwbextractor.

        Adds traces from recording object as ElectricalSeries to nwbfile object.

        Parameters
        ----------
        recording: RecordingExtractor
        nwbfile: NWBFile
            nwb file to which the recording information is to be added
        metadata: dict
            metadata info for constructing the nwb file (optional).
            Should be of the format
                metadata['Ecephys']['ElectricalSeries'] = {'name': my_name,
                                                            'description': my_description}
        buffer_mb: int (optional, defaults to 500MB)
            maximum amount of memory (in MB) to use per iteration of the
            DataChunkIterator (requires traces to be memmap objects)
        use_times: bool (optional, defaults to False)
            If True, the times are saved to the nwb file using recording.frame_to_time(). If False (defualut),
            the sampling rate is used.
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
        iterate: bool (optional, defaults to True)
            Whether or not to use DataChunkIteration. Highly recommended for large (16+ GB) recordings.

        Missing keys in an element of metadata['Ecephys']['ElectrodeGroup'] will be auto-populated with defaults
        whenever possible.
        """
        if self.nwbfile is not None:
            assert isinstance(self.nwbfile, pynwb.NWBFile), "'nwbfile' should be of type pynwb.NWBFile!"
        assert (
            self._conversion_ops["buffer_mb"] > 10
        ), "'buffer_mb' should be at least 10MB to ensure data can be chunked!"

        if not self.nwbfile.electrodes:
            self.add_electrodes()

        if self._conversion_ops["compression"] == "lzf" and self._conversion_ops["compression_opts"] is not None:
            warn(
                f"compression_opts ({self._conversion_ops['compression_opts']})"
                "were passed, but compression type is 'lzf'! Ignoring options."
            )
            compression_opts = None

        if self._conversion_ops["write_as"] == "raw":
            eseries_kwargs = dict(
                name="ElectricalSeries_raw",
                description="Raw acquired data",
                comments="Generated from SpikeInterface::NwbRecordingExtractor",
            )
        elif self._conversion_ops["write_as"] == "processed":
            eseries_kwargs = dict(
                name="ElectricalSeries_processed",
                description="Processed data",
                comments="Generated from SpikeInterface::NwbRecordingExtractor",
            )
            # Check for existing processing module and data interface
            ecephys_mod = check_module(
                nwbfile=self.nwbfile,
                name="ecephys",
                description="Intermediate data from extracellular electrophysiology recordings, e.g., LFP.",
            )
            if "Processed" not in ecephys_mod.data_interfaces:
                ecephys_mod.add(pynwb.ecephys.FilteredEphys(name="Processed"))
        elif self._conversion_ops["write_as"] == "lfp":
            eseries_kwargs = dict(
                name="ElectricalSeries_lfp",
                description="Processed data - LFP",
                comments="Generated from SpikeInterface::NwbRecordingExtractor",
            )
            # Check for existing processing module and data interface
            ecephys_mod = check_module(
                nwbfile=self.nwbfile,
                name="ecephys",
                description="Intermediate data from extracellular electrophysiology recordings, e.g., LFP.",
            )
            if "LFP" not in ecephys_mod.data_interfaces:
                ecephys_mod.add(pynwb.ecephys.LFP(name="LFP"))

        # If user passed metadata info, overwrite defaults
        if self.metadata is not None and "Ecephys" in self.metadata:
            assert (
                self._conversion_ops["es_key"] in self.metadata["Ecephys"]
            ), f"metadata['Ecephys'] dictionary does not contain key '{self._conversion_ops['es_key']}'"
            eseries_kwargs.update(self.metadata["Ecephys"][self._conversion_ops["es_key"]])

        # Check for existing names in nwbfile
        if self._conversion_ops["write_as"] == "raw":
            assert (
                eseries_kwargs["name"] not in self.nwbfile.acquisition
            ), f"Raw ElectricalSeries '{eseries_kwargs['name']}' is already written in the NWBFile!"
        elif self._conversion_ops["write_as"] == "processed":
            assert (
                eseries_kwargs["name"]
                not in self.nwbfile.processing["ecephys"].data_interfaces["Processed"].electrical_series
            ), f"Processed ElectricalSeries '{eseries_kwargs['name']}' is already written in the NWBFile!"
        elif self._conversion_ops["write_as"] == "lfp":
            assert (
                eseries_kwargs["name"]
                not in self.nwbfile.processing["ecephys"].data_interfaces["LFP"].electrical_series
            ), f"LFP ElectricalSeries '{eseries_kwargs['name']}' is already written in the NWBFile!"

        # Electrodes table region
        channel_ids = self._get_channel_ids()
        table_ids = [list(self.nwbfile.electrodes.id[:]).index(id) for id in channel_ids]
        electrode_table_region = self.nwbfile.create_electrode_table_region(
            region=table_ids, description="electrode_table_region"
        )
        eseries_kwargs.update(electrodes=electrode_table_region)

        # channels gains - for RecordingExtractor, these are values to cast traces to uV.
        # For nwb, the conversions (gains) cast the data to Volts.
        # To get traces in Volts we take data*channel_conversion*conversion.
        channel_conversion = self._get_channel_property_values("gain", channel_ids)
        channel_offset = self._get_channel_property_values("offset", channel_ids)
        unsigned_coercion = channel_offset / channel_conversion
        if not np.all([x.is_integer() for x in unsigned_coercion]):
            raise NotImplementedError(
                "Unable to coerce underlying unsigned data type to signed type, which is currently required for NWB "
                "Schema v2.2.5! Please specify 'write_scaled=True'."
            )
        elif np.any(unsigned_coercion != 0):
            warnings.warn(
                "NWB Schema v2.2.5 does not officially support channel offsets. The data will be converted to a signed "
                "type that does not use offsets."
            )
            unsigned_coercion = unsigned_coercion.astype(int)
        if self._conversion_ops["write_scaled"]:
            eseries_kwargs.update(conversion=1e-6)
        else:
            if len(np.unique(channel_conversion)) == 1:  # if all gains are equal
                eseries_kwargs.update(conversion=channel_conversion[0] * 1e-6)
            else:
                eseries_kwargs.update(conversion=1e-6)
                eseries_kwargs.update(channel_conversion=channel_conversion)

        trace_dtype = self._get_traces(channel_ids=channel_ids[:1], end_frame=1).dtype
        estimated_memory = trace_dtype.itemsize * len(self._get_channel_ids()) * self._get_num_frames()
        if not self._conversion_ops["iterate"] and psutil.virtual_memory().available <= estimated_memory:
            warn("iteration was disabled, but not enough memory to load traces! Forcing iterate=True.")
            iterate = True
        if self._conversion_ops["iterate"]:
            if isinstance(
                self._get_traces(end_frame=5, return_scaled=self._conversion_ops["write_scaled"]), np.memmap
            ) and np.all(channel_offset == 0):
                n_bytes = np.dtype(self.recording.get_dtype()).itemsize
                buffer_size = int(self._conversion_ops["buffer_mb"] * 1e6) // (len(self._get_channel_ids()) * n_bytes)
                ephys_data = DataChunkIterator(
                    data=self._get_traces(return_scaled=self._conversion_ops["write_scaled"]).T,
                    # nwb standard is time as zero axis
                    buffer_size=buffer_size,
                )
            else:

                def data_generator(recording, channels_ids, unsigned_coercion, write_scaled):
                    for i, ch in enumerate(channels_ids):
                        data = recording._get_traces(channel_ids=[ch], return_scaled=write_scaled)
                        if not write_scaled:
                            data_dtype_name = data.dtype.name
                            if data_dtype_name.startswith("uint"):
                                data_dtype_name = data_dtype_name[1:]  # Retain memory of signed data type
                            data = data + unsigned_coercion[i]
                            data = data.astype(data_dtype_name)
                        yield data.flatten()

                ephys_data = DataChunkIterator(
                    data=data_generator(
                        recording=self,
                        channels_ids=channel_ids,
                        unsigned_coercion=unsigned_coercion,
                        write_scaled=self._conversion_ops["write_scaled"],
                    ),
                    iter_axis=1,  # nwb standard is time as zero axis
                    maxshape=(self._get_num_frames(), len(self._get_channel_ids())),
                )
        else:
            ephys_data = self._get_traces(return_scaled=self._conversion_ops["write_scaled"]).T

        eseries_kwargs.update(
            data=H5DataIO(
                ephys_data,
                compression=self._conversion_ops["compression"],
                compression_opts=self._conversion_ops["compression_opts"],
            )
        )
        if not self._conversion_ops["use_times"]:
            eseries_kwargs.update(
                starting_time=float(self._get_times()[0]),
                rate=float(self._get_sampling_frequency()),
            )
        else:
            eseries_kwargs.update(
                timestamps=H5DataIO(
                    self._get_times(),
                    compression=self._conversion_ops["compression"],
                    compression_opts=self._conversion_ops["compression_opts"],
                )
            )

        # Add ElectricalSeries to nwbfile object
        es = pynwb.ecephys.ElectricalSeries(**eseries_kwargs)
        if self._conversion_ops["write_as"] == "raw":
            self.nwbfile.add_acquisition(es)
        elif self._conversion_ops["write_as"] == "processed":
            ecephys_mod.data_interfaces["Processed"].add_electrical_series(es)
        elif self._conversion_ops["write_as"] == "lfp":
            ecephys_mod.data_interfaces["LFP"].add_electrical_series(es)

    def add_units(self):
        """Auxilliary function for add_sorting."""
        unit_ids = self._get_unit_ids()
        fs = self._get_unit_sampling_frequency()
        if fs is None:
            raise ValueError("Writing a SortingExtractor to an NWBFile requires a known sampling frequency!")

        all_properties = set()
        all_features = set()
        for unit_id in unit_ids:
            all_properties.update(self._get_unit_property_names(unit_id))
            all_features.update(self._get_unit_feature_names(unit_id))

        if self._conversion_ops["unit_property_descriptions"] is None:
            property_descriptions = dict(_default_sorting_property_descriptions)
        else:
            property_descriptions = dict(
                _default_sorting_property_descriptions, **self._conversion_ops["unit_property_descriptions"]
            )

        if self.nwbfile.units is None:
            # Check that array properties have the same shape across units
            property_shapes = dict()
            for pr in all_properties:
                shapes = []
                for unit_id in unit_ids:
                    if pr in self._get_unit_property_names(unit_id):
                        prop_value = self._get_unit_property_values(pr, unit_id)
                        if isinstance(prop_value, (int, np.integer, float, str, bool)):
                            shapes.append(1)
                        elif isinstance(prop_value, (list, np.ndarray)):
                            if np.array(prop_value).ndim == 1:
                                shapes.append(len(prop_value))
                            else:
                                shapes.append(np.array(prop_value).shape)
                        elif isinstance(prop_value, dict):
                            print(f"Skipping property '{pr}' because dictionaries are not supported.")
                            self._conversion_ops["skip_unit_properties"].append(pr)
                            break
                    else:
                        shapes.append(np.nan)
                property_shapes[pr] = shapes

            for pr in property_shapes.keys():
                elems = [elem for elem in property_shapes[pr] if not np.any(np.isnan(elem))]
                if not np.all([elem == elems[0] for elem in elems]):
                    print(f"Skipping property '{pr}' because it has variable size across units.")
                    self._conversion_ops["skip_unit_properties"].append(pr)

            write_properties = set(all_properties) - set(self._conversion_ops["skip_unit_properties"])
            for pr in write_properties:
                if pr not in property_descriptions:
                    warnings.warn(
                        f"Description for property {pr} not found in property_descriptions. "
                        f"Description for property {pr} not found in property_descriptions. "
                        "Setting description to 'no description'"
                    )
            for pr in write_properties:
                unit_col_args = dict(name=pr, description=property_descriptions.get(pr, "No description."))
                if pr in ["max_channel", "max_electrode"] and self.nwbfile.electrodes is not None:
                    unit_col_args.update(table=self.nwbfile.electrodes)
                self.nwbfile.add_unit_column(**unit_col_args)

            for unit_id in unit_ids:
                unit_kwargs = dict()
                if self._conversion_ops["use_times"]:
                    spkt = self._get_unit_spike_train_times(unit_id)
                else:
                    spkt = self._get_unit_spike_train_ids(unit_id) / self._get_unit_sampling_frequency()
                for pr in write_properties:
                    if pr in self._get_unit_property_names(unit_id):
                        prop_value = self._get_unit_property_values(pr, unit_id)
                        unit_kwargs.update({pr: prop_value})
                    else:  # Case of missing data for this unit and this property
                        unit_kwargs.update({pr: np.nan})
                self.nwbfile.add_unit(id=int(unit_id), spike_times=spkt, **unit_kwargs)

            # Check that multidimensional features have the same shape across units
            feature_shapes = dict()
            for ft in all_features:
                shapes = []
                for unit_id in unit_ids:
                    if ft in self._get_unit_feature_names(unit_id):
                        feat_value = self._get_unit_feature_values(ft, unit_id)
                        if isinstance(feat_value[0], (int, np.integer, float, str, bool)):
                            break
                        elif isinstance(feat_value[0], (list, np.ndarray)):  # multidimensional features
                            if np.array(feat_value).ndim > 1:
                                shapes.append(np.array(feat_value).shape)
                                feature_shapes[ft] = shapes
                        elif isinstance(feat_value[0], dict):
                            print(f"Skipping feature '{ft}' because dictionaries are not supported.")
                            self._conversion_ops["skip_unit_features"].append(ft)
                            break
                    else:
                        print(f"Skipping feature '{ft}' because not share across all units.")
                        self._conversion_ops["skip_unit_features"].append(ft)
                        break

            nspikes = {k: get_num_spikes(self.nwbfile.units, int(k)) for k in unit_ids}

            for ft in feature_shapes.keys():
                # skip first dimension (num_spikes) when comparing feature shape
                if not np.all([elem[1:] == feature_shapes[ft][0][1:] for elem in feature_shapes[ft]]):
                    print(f"Skipping feature '{ft}' because it has variable size across units.")
                    self._conversion_ops["skip_unit_features"].append(ft)

            for ft in set(all_features) - set(self._conversion_ops["skip_unit_features"]):
                values = []
                if not ft.endswith("_idxs"):
                    for unit_id in self._get_unit_ids():
                        feat_vals = self._get_unit_feature_values(ft, unit_id)

                        if len(feat_vals) < nspikes[unit_id]:
                            self._conversion_ops["skip_unit_features"].append(ft)
                            print(f"Skipping feature '{ft}' because it is not defined for all spikes.")
                            break
                            # this means features are available for a subset of spikes
                            # all_feat_vals = np.array([np.nan] * nspikes[unit_id])
                            # feature_idxs = sorting.get_unit_spike_features(unit_id, feat_name + '_idxs')
                            # all_feat_vals[feature_idxs] = feat_vals
                        else:
                            all_feat_vals = feat_vals
                        values.append(all_feat_vals)

                    flatten_vals = [item for sublist in values for item in sublist]
                    nspks_list = [sp for sp in nspikes.values()]
                    spikes_index = np.cumsum(nspks_list).astype("int64")
                    if ft in self.nwbfile.units:  # If property already exists, skip it
                        warnings.warn(f"Feature {ft} already present in units table, skipping it")
                        continue
                    set_dynamic_table_property(
                        dynamic_table=self.nwbfile.units,
                        row_ids=[int(k) for k in unit_ids],
                        property_name=ft,
                        values=flatten_vals,
                        index=spikes_index,
                    )
        else:
            warnings.warn("The nwbfile already contains units. These units will not be over-written.")

    @abstractmethod
    def add_epochs(self):
        pass


def list_get(li: list, idx: int, default):
    """Safe index retrieval from list."""
    try:
        return li[idx]
    except IndexError:
        return default


def set_dynamic_table_property(
    dynamic_table,
    row_ids,
    property_name,
    values,
    index=False,
    default_value=np.nan,
    table=False,
    description="no description",
):
    if not isinstance(row_ids, list) or not all(isinstance(x, int) for x in row_ids):
        raise TypeError("'ids' must be a list of integers")
    ids = list(dynamic_table.id[:])
    if any([i not in ids for i in row_ids]):
        raise ValueError("'ids' contains values outside the range of existing ids")
    if not isinstance(property_name, str):
        raise TypeError("'property_name' must be a string")
    if len(row_ids) != len(values) and index is False:
        raise ValueError("'ids' and 'values' should be lists of same size")

    if index is False:
        if property_name in dynamic_table:
            for (row_id, value) in zip(row_ids, values):
                dynamic_table[property_name].data[ids.index(row_id)] = value
        else:
            col_data = [default_value] * len(ids)  # init with default val
            for (row_id, value) in zip(row_ids, values):
                col_data[ids.index(row_id)] = value
            dynamic_table.add_column(
                name=property_name, description=description, data=col_data, index=index, table=table
            )
    else:
        if property_name in dynamic_table:
            # TODO
            raise NotImplementedError
        else:
            dynamic_table.add_column(name=property_name, description=description, data=values, index=index, table=table)


def check_module(nwbfile, name: str, description: str = None):
    """
    Check if processing module exists. If not, create it. Then return module.

    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    name: str
    description: str | None (optional)

    Returns
    -------
    pynwb.module
    """
    assert isinstance(nwbfile, pynwb.NWBFile), "'nwbfile' should be of type pynwb.NWBFile"
    if name in nwbfile.modules:
        return nwbfile.modules[name]
    else:
        if description is None:
            description = name
        return nwbfile.create_processing_module(name, description)


def get_num_spikes(units_table, unit_id):
    """Return the number of spikes for chosen unit."""
    ids = np.array(units_table.id[:])
    indexes = np.where(ids == unit_id)[0]
    if not len(indexes):
        raise ValueError(f"{unit_id} is an invalid unit_id. Valid ids: {ids}.")
    index = indexes[0]
    if index == 0:
        return units_table["spike_times_index"].data[index]
    else:
        return units_table["spike_times_index"].data[index] - units_table["spike_times_index"].data[index - 1]
