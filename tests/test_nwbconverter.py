from nwb_conversion_tools import dynamic_nwb_converter, NeuroscopeRecordingInterface, CellExplorerSortingInterface


def test_dynamic_nwb_converter():

    dynamic_nwb_converter(
        data_interface_classes=dict(
            neuroscope=NeuroscopeRecordingInterface,
            cell_explorer=CellExplorerSortingInterface
        ),
        name='CustomNWBConverter1'
    )
