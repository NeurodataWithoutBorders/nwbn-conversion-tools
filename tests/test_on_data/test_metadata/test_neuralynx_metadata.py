from pathlib import Path
import datetime

from nwb_conversion_tools.datainterfaces.ecephys.neuralynx.neuralynxdatainterface import (
    get_metadata,
    get_filtering,
    NeuralynxRecordingInterface,
)

from ..setup_paths import ECEPHY_DATA_PATH

# NLX_PATH = ECEPHY_DATA_PATH / "neuralynx"


def test_neuralynx_cheetah_v574_metadata():
    NLX_PATH = ECEPHY_DATA_PATH / "neuralynx"
    folder_path = NLX_PATH / "Cheetah_v5.7.4" / "original_data"
    assert folder_path.is_dir()
    assert get_metadata(str(folder_path)) == dict(
        session_start_time=datetime.datetime(2017, 2, 16, 17, 56, 4), session_id="d8ba8eef-8d11-4cdc-86dc-05f50d4ba13d"
    )


def test_neuralynx_cheetah_v563_metadata():
    NLX_PATH = ECEPHY_DATA_PATH / "neuralynx"
    folder_path = NLX_PATH / "Cheetah_v5.6.3" / "original_data"
    assert folder_path.is_dir()
    assert get_metadata(str(folder_path)) == dict(
        session_start_time=datetime.datetime(2016, 11, 28, 21, 50, 33, 322000)
    )


def test_neuralynx_cheetah_v540_metadata():
    NLX_PATH = ECEPHY_DATA_PATH / "neuralynx"
    folder_path = NLX_PATH / "Cheetah_v5.4.0" / "original_data"
    assert folder_path.is_dir()
    assert get_metadata(str(folder_path)) == dict(
        session_start_time=datetime.datetime(2001, 1, 1, 0, 0),
    )


def test_neuralynx_filtering():
    NLX_PATH = ECEPHY_DATA_PATH / "neuralynx"
    file_path = NLX_PATH / "Cheetah_v5.7.4" / "original_data" / "CSC1.ncs"
    assert file_path.is_file()
    assert (
        get_filtering(str(file_path)) == '{"DSPLowCutFilterEnabled": "True", '
        '"DspLowCutFrequency": "10", '
        '"DspLowCutNumTaps": "0", '
        '"DspLowCutFilterType": "DCO", '
        '"DSPHighCutFilterEnabled": "True", '
        '"DspHighCutFrequency": "9000", '
        '"DspHighCutNumTaps": "64", '
        '"DspHighCutFilterType": "FIR", '
        '"DspDelayCompensation": "Enabled", '
        '"DspFilterDelay_µs": "984"}'
    )


def test_neuralynx_filtering_recording_extractor():
    NLX_PATH = ECEPHY_DATA_PATH / "neuralynx"

    ri = NeuralynxRecordingInterface(folder_path=(NLX_PATH / "Cheetah_v5.7.4" / "original_data"))

    assert (
        ri.recording_extractor.get_channel_property(0, "filtering") == '{"DSPLowCutFilterEnabled": "True", '
        '"DspLowCutFrequency": "10", '
        '"DspLowCutNumTaps": "0", '
        '"DspLowCutFilterType": "DCO", '
        '"DSPHighCutFilterEnabled": "True", '
        '"DspHighCutFrequency": "9000", '
        '"DspHighCutNumTaps": "64", '
        '"DspHighCutFilterType": "FIR", '
        '"DspDelayCompensation": "Enabled", '
        '"DspFilterDelay_µs": "984"}'
    )
