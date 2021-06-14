from unittest import TestCase

from pynwb.base import ProcessingModule

from nwb_conversion_tools.utils.conversion_tools import check_regular_timestamps, get_module, make_nwbfile_from_metadata


class TestConversionTools(TestCase):

    def test_check_regular_timestamps(self):
        assert check_regular_timestamps([1, 2, 3])
        assert not check_regular_timestamps([1, 2, 4])

    def test_get_module(self):
        nwbfile = make_nwbfile_from_metadata(metadata=dict())

        name_1 = "test_1"
        name_2 = "test_2"
        description_1 = "description_1"
        description_2 = "description_2"
        nwbfile.create_processing_module(name=name_1, description=description_1)
        mod_1 = get_module(nwbfile=nwbfile, name=name_1, description=description_1)
        mod_2 = get_module(nwbfile=nwbfile, name=name_2, description=description_2)
        assert isinstance(mod_1, ProcessingModule)
        assert mod_1.description == description_1
        assert isinstance(mod_2, ProcessingModule)
        assert mod_2.description == description_2
        self.assertWarns(UserWarning, get_module, **dict(nwbfile=nwbfile, name=name_1, description=description_2))
