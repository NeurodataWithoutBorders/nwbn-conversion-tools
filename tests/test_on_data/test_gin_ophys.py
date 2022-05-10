import unittest
from datetime import datetime

from parameterized import parameterized, param
from roiextractors import NwbImagingExtractor, NwbSegmentationExtractor
from roiextractors.testing import check_imaging_equal, check_segmentations_equal

from nwb_conversion_tools import (
    NWBConverter,
    TiffImagingInterface,
    Hdf5ImagingInterface,
    SbxImagingInterface,
    CaimanSegmentationInterface,
    CnmfeSegmentationInterface,
    ExtractSegmentationInterface,
    Suite2pSegmentationInterface,
)

from .setup_paths import OPHYS_DATA_PATH, OUTPUT_PATH


def custom_name_func(testcase_func, param_num, param):
    return (
        f"{testcase_func.__name__}_{param_num}_"
        f"{parameterized.to_safe_name(param.kwargs['data_interface'].__name__)}"
    )


class TestOphysNwbConversions(unittest.TestCase):
    savedir = OUTPUT_PATH

    imaging_interface_list = [
        param(
            data_interface=TiffImagingInterface,
            interface_kwargs=dict(
                file_path=str(OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "demoMovie.tif"),
                sampling_frequency=15.0,  # typically provied by user
            ),
        ),
        param(
            data_interface=Hdf5ImagingInterface,
            interface_kwargs=dict(file_path=str(OPHYS_DATA_PATH / "imaging_datasets" / "hdf5" / "demoMovie.hdf5")),
        ),
    ]
    for suffix in [".mat", ".sbx"]:
        imaging_interface_list.append(
            param(
                data_interface=SbxImagingInterface,
                interface_kwargs=dict(
                    file_path=str(OPHYS_DATA_PATH / "imaging_datasets" / "Scanbox" / f"sample{suffix}")
                ),
            ),
        )

    @parameterized.expand(imaging_interface_list, name_func=custom_name_func)
    def test_convert_imaging_extractor_to_nwb(self, data_interface, interface_kwargs):
        nwbfile_path = self.savedir / f"{data_interface.__name__}.nwb"

        # TODO: Temporary hack around a strange issue where if the first SBX file fails due to an error
        # during check_imaging_equal, it leaves the NWBFile open and second test fails because of that.
        # Try to determine true source of error; is context failing to close through pynwb or is it the
        # NWBImagingExtractor that fails to close?
        if nwbfile_path.exists():
            nwbfile_path = self.savedir / f"{data_interface.__name__}_2.nwb"

        class TestConverter(NWBConverter):
            data_interface_classes = dict(TestImaging=data_interface)

            def get_metadata(self):
                metadata = super().get_metadata()
                # attach device to ImagingPlane lacking property
                device_name = metadata["Ophys"]["Device"][0]["name"]
                if "device" not in metadata["Ophys"]["ImagingPlane"][0].keys():
                    metadata["Ophys"]["ImagingPlane"][0]["device"] = device_name
                # attach ImagingPlane to TwoPhotonSeries lacking property
                plane_name = metadata["Ophys"]["ImagingPlane"][0]["name"]
                if "imaging_plane" not in metadata["Ophys"]["TwoPhotonSeries"][0].keys():
                    metadata["Ophys"]["TwoPhotonSeries"][0]["imaging_plane"] = plane_name
                return metadata

        converter = TestConverter(source_data=dict(TestImaging=dict(interface_kwargs)))
        metadata = converter.get_metadata()
        metadata["NWBFile"].update(session_start_time=datetime.now().astimezone())
        converter.run_conversion(nwbfile_path=nwbfile_path, overwrite=True, metadata=metadata)
        imaging = converter.data_interface_objects["TestImaging"].imaging_extractor
        nwb_imaging = NwbImagingExtractor(file_path=nwbfile_path)
        check_imaging_equal(img1=imaging, img2=nwb_imaging)

    @parameterized.expand(
        [
            param(
                data_interface=CaimanSegmentationInterface,
                interface_kwargs=dict(
                    file_path=str(OPHYS_DATA_PATH / "segmentation_datasets" / "caiman" / "caiman_analysis.hdf5")
                ),
            ),
            param(
                data_interface=CnmfeSegmentationInterface,
                interface_kwargs=dict(
                    file_path=str(
                        OPHYS_DATA_PATH
                        / "segmentation_datasets"
                        / "cnmfe"
                        / "2014_04_01_p203_m19_check01_cnmfeAnalysis.mat"
                    )
                ),
            ),
            param(
                data_interface=ExtractSegmentationInterface,
                interface_kwargs=dict(
                    file_path=str(
                        OPHYS_DATA_PATH
                        / "segmentation_datasets"
                        / "extract"
                        / "2014_04_01_p203_m19_check01_extractAnalysis.mat"
                    )
                ),
            ),
            param(
                data_interface=Suite2pSegmentationInterface,
                interface_kwargs=dict(
                    # TODO: argument name is 'file_path' on roiextractors, but it clearly refers to a folder_path
                    file_path=str(OPHYS_DATA_PATH / "segmentation_datasets" / "suite2p")
                ),
            ),
        ],
        name_func=custom_name_func,
    )
    def test_convert_segmentation_extractor_to_nwb(self, data_interface, interface_kwargs):
        nwbfile_path = str(self.savedir / f"{data_interface.__name__}.nwb")

        class TestConverter(NWBConverter):
            data_interface_classes = dict(TestSegmentation=data_interface)

        converter = TestConverter(source_data=dict(TestSegmentation=dict(interface_kwargs)))
        metadata = converter.get_metadata()
        metadata["NWBFile"].update(session_start_time=datetime.now().astimezone())
        converter.run_conversion(nwbfile_path=nwbfile_path, overwrite=True, metadata=metadata)
        segmentation = converter.data_interface_objects["TestSegmentation"].segmentation_extractor
        nwb_segmentation = NwbSegmentationExtractor(file_path=nwbfile_path)
        check_segmentations_equal(seg1=segmentation, seg2=nwb_segmentation)


if __name__ == "__main__":
    unittest.main()
