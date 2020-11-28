"""Authors: Cody Baker and Ben Dichter."""
from abc import ABC

import spikeextractors as se
import numpy as np
from pynwb import NWBFile
from pynwb.ecephys import SpikeEventSeries

from .basedatainterface import BaseDataInterface
from .utils import get_schema_from_hdmf_class, add_ecephys_metadata
from .json_schema_utils import get_base_schema, get_schema_from_method_signature, fill_defaults


class BaseSortingExtractorInterface(BaseDataInterface, ABC):
    SX = None

    @classmethod
    def get_source_schema(cls):
        return get_schema_from_method_signature(cls.SX.__init__)

    def __init__(self, **source_data):
        super().__init__(**source_data)
        self.sorting_extractor = self.SX(**source_data)

    def get_metadata_schema(self):
        metadata_schema = get_base_schema(
            required=['SpikeEventSeries'],
            properties=dict(
                SpikeEventSeries=get_schema_from_hdmf_class(SpikeEventSeries)
            )
        )
        fill_defaults(metadata_schema, self.get_metadata())

        return metadata_schema

    def run_conversion(self, nwbfile: NWBFile, metadata: dict, stub_test: bool = False,
                       write_ecephys_metadata: bool = False):
        if 'UnitProperties' not in metadata:
            metadata['UnitProperties'] = []
        if write_ecephys_metadata:
            add_ecephys_metadata(nwbfile, metadata)

        property_descriptions = dict()
        if stub_test:
            max_min_spike_time = max([min(x) for y in self.sorting_extractor.get_unit_ids()
                                      for x in [self.sorting_extractor.get_unit_spike_train(y)] if any(x)])
            stub_sorting_extractor = se.SubSortingExtractor(
                self.sorting_extractor,
                unit_ids=self.sorting_extractor.get_unit_ids(),
                start_frame=0,
                end_frame=1.1 * max_min_spike_time
            )
            sorting_extractor = stub_sorting_extractor
        else:
            sorting_extractor = self.sorting_extractor

        for metadata_column in metadata['UnitProperties']:
            property_descriptions.update({metadata_column['name']: metadata_column['description']})
            for unit_id in sorting_extractor.get_unit_ids():
                if metadata_column['name'] == 'electrode_group':
                    data = nwbfile.electrode_groups[metadata_column['data'][unit_id]]
                else:
                    data = metadata_column['data'][unit_id]
                sorting_extractor.set_unit_property(unit_id, metadata_column['name'], data)

        se.NwbSortingExtractor.write_sorting(
            sorting_extractor,
            property_descriptions=property_descriptions,
            nwbfile=nwbfile
        )
