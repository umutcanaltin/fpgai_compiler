from fpgai.implementations.implementation_contract import ImplementationContract
from fpgai.implementations.hls_integration import parse_tensor_ports_abi


def _contract():
    return ImplementationContract(
        package_id='community.hetero_hls', version='1.0.0', operator_id='community.operator.hetero',
        language='hls_cpp', backend='vitis_hls', top='hetero', sources=('src/hetero.cpp',),
        validation_level='reference_tested',
        metadata={'integration':{'hls':{
            'abi':'tensor_ports_v1','scalar_type':'float','count_mode':'per_port',
            'inputs':[{'name':'a','shape':[1,4],'layout':'NC'},{'name':'b','scalar_type':'double','shape':[1,2]}],
            'outputs':[{'name':'out','shape':[1,3]}],
        }}}
    )


def test_tensor_ports_support_port_metadata_and_per_port_counts():
    abi=parse_tensor_ports_abi(_contract())
    assert abi.count_mode=='per_port'
    assert abi.inputs[0].shape==(1,4)
    assert abi.inputs[0].layout=='NC'
    assert abi.scalar_for(abi.inputs[1])=='double'
