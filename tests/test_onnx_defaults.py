import pytest
from fpgai_research.frontend.attributes import canonical_attributes

def test_exporter_defaults_are_semantically_neutral():
    assert canonical_attributes("MaxPool",{
        "kernel_shape":[2,2],"strides":[2,2],"auto_pad":b"NOTSET",
        "ceil_mode":0,"storage_order":0,"dilations":[1,1]})=={
        "kernel_shape":(2,2),"strides":(2,2)}
    assert canonical_attributes("Conv",{
        "pads":[0,0,0,0],"auto_pad":b"NOTSET","group":1})=={
        "pads":(0,0,0,0),"group":1}

@pytest.mark.parametrize("kind,change",[
    ("MaxPool",{"ceil_mode":1}),("MaxPool",{"dilations":[2,2]}),
    ("MaxPool",{"storage_order":1}),("Conv",{"auto_pad":b"SAME_UPPER"}),
    ("Conv",{"foo":1})])
def test_unsupported_semantics_rejected(kind,change):
    with pytest.raises(ValueError):
        canonical_attributes(kind,change)
