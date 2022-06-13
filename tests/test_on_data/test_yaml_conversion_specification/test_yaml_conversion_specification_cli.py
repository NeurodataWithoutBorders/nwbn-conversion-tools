from pathlib import Path
from datetime import datetime

from hdmf.testing import TestCase
from pynwb import NWBHDF5IO

from nwb_conversion_tools.tools.data_transfers import deploy_process

from .setup_paths import ECEPHY_DATA_PATH as DATA_PATH
from .setup_paths import OUTPUT_PATH


class TestYAMLConversionSpecification(TestCase):
    test_folder = OUTPUT_PATH

    def test_run_conversion_from_yaml_cli(self):
        path_to_test_yml_files = Path(__file__).parent / "conversion_specifications"
        yaml_file_path = path_to_test_yml_files / "GIN_conversion_specification.yml"
        deploy_process(
            command=(
                f"nwbct-run-conversion {yaml_file_path} "
                f"--data-folder {DATA_PATH} --output-path {self.test_folder} --overwrite"
            )
        )

        with NWBHDF5IO(path=self.test_folder / "example_converter_spec_1.nwb", mode="r") as io:
            nwbfile = io.read()
            assert nwbfile.session_description == "Subject navigating a Y-shaped maze."
            assert nwbfile.lab == "My Lab"
            assert nwbfile.institution == "My Institution"
            assert nwbfile.session_start_time == datetime.fromisoformat("2020-10-09T21:19:09+00:00")
            assert nwbfile.subject.subject_id == "1"
            assert "ElectricalSeries_raw" in nwbfile.acquisition
        with NWBHDF5IO(path=self.test_folder / "example_converter_spec_2.nwb", mode="r") as io:
            nwbfile = io.read()
            assert nwbfile.session_description == "Subject navigating a Y-shaped maze."
            assert nwbfile.lab == "My Lab"
            assert nwbfile.institution == "My Institution"
            assert nwbfile.session_start_time == datetime.fromisoformat("2020-10-10T21:19:09+00:00")
            assert nwbfile.subject.subject_id == "002"
        with NWBHDF5IO(path=self.test_folder / "example_converter_spec_3.nwb", mode="r") as io:
            nwbfile = io.read()
            assert nwbfile.session_description == "no description"
            assert nwbfile.lab == "My Lab"
            assert nwbfile.institution == "My Institution"
            assert nwbfile.session_start_time == datetime.fromisoformat("2020-10-11T21:19:09+00:00")
            assert nwbfile.subject.subject_id == "Subject Name"
            assert "spike_times" in nwbfile.units

    def test_run_conversion_from_yaml_default_nwbfile_name(self):
        self.test_folder = self.test_folder / "test_organize"
        self.test_folder.mkdir(exist_ok=True)
        path_to_test_yml_files = Path(__file__).parent / "conversion_specifications"
        yaml_file_path = path_to_test_yml_files / "GIN_conversion_specification_missing_nwbfile_names.yml"
        deploy_process(
            command=(
                f"nwbct-run-conversion {yaml_file_path} "
                f"--data-folder {DATA_PATH} --output-path {self.test_folder} --overwrite"
            )
        )

        with NWBHDF5IO(path=self.test_folder / "sub-Mouse_1_ses-20201009T211909.nwb", mode="r") as io:
            nwbfile = io.read()
            assert nwbfile.session_description == "Subject navigating a Y-shaped maze."
            assert nwbfile.lab == "My Lab"
            assert nwbfile.institution == "My Institution"
            assert nwbfile.session_start_time == datetime.fromisoformat("2020-10-09T21:19:09+00:00")
            assert nwbfile.subject.subject_id == "Mouse 1"
            assert "ElectricalSeries_raw" in nwbfile.acquisition
        with NWBHDF5IO(path=self.test_folder / "example_defined_name.nwb", mode="r") as io:
            nwbfile = io.read()
            assert nwbfile.session_description == "Subject navigating a Y-shaped maze."
            assert nwbfile.lab == "My Lab"
            assert nwbfile.institution == "My Institution"
            assert nwbfile.session_start_time == datetime.fromisoformat("2020-10-10T21:19:09+00:00")
            assert nwbfile.subject.subject_id == "MyMouse002"
        with NWBHDF5IO(path=self.test_folder / "sub-Subject_Name_ses-20201011T211909.nwb", mode="r") as io:
            nwbfile = io.read()
            assert nwbfile.session_description == "no description"
            assert nwbfile.lab == "My Lab"
            assert nwbfile.institution == "My Institution"
            assert nwbfile.session_start_time == datetime.fromisoformat("2020-10-11T21:19:09+00:00")
            assert nwbfile.subject.subject_id == "Subject Name"
            assert "spike_times" in nwbfile.units