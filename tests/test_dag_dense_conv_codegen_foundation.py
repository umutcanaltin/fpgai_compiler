from types import SimpleNamespace
import numpy as np
from fpgai.backends.hls.buffer_allocation import build_hls_buffer_allocation
from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.ir.graph import Graph, Op, TensorSpec


def _cfg():
    return {"numerics":{"kind":"fixed","defaults":{"activation":{"type":"ap_fixed","total_bits":16,"int_bits":6}}}}


def test_dag_dense_codegen_reuses_existing_dense_kernel():
    g=Graph(name="dense_branch"); g.inputs=["x"]; g.outputs=["out"]
    g.tensors["x"]=TensorSpec(name="x",shape=(1,4),dtype="float32")
    g.constants["W"]=np.ones((4,4),dtype=np.float32); g.constants["B"]=np.zeros((4,),dtype=np.float32)
    g.tensors["W"]=TensorSpec(name="W",shape=(4,4),dtype="float32"); g.tensors["B"]=TensorSpec(name="B",shape=(4,),dtype="float32")
    g.ops=[Op("Relu","relu",["x"],["r"],{}),Op("Dense","dense",["x","W","B"],["d"],{"in_features":4,"out_features":4}),Op("Add","add",["r","d"],["out"],{})]
    for n in ["r","d","out"]: g.tensors[n]=TensorSpec(name=n,shape=(1,4),dtype="float32")
    alloc=build_hls_buffer_allocation(g,raw_cfg=_cfg())
    src=emit_dag_top_cpp(g,top_name="deeplearn",weights_mode="embedded",raw_cfg=_cfg(),buffer_allocation=alloc)
    assert "dense_out_in<" in src
    assert "W0, B0" in src


def test_dag_conv_codegen_reuses_existing_conv_kernel():
    g=Graph(name="conv_branch"); g.inputs=["x"]; g.outputs=["out"]
    g.tensors["x"]=TensorSpec(name="x",shape=(1,1,4,4),dtype="float32")
    g.constants["W"]=np.ones((1,1,3,3),dtype=np.float32); g.constants["B"]=np.zeros((1,),dtype=np.float32)
    g.tensors["W"]=TensorSpec(name="W",shape=(1,1,3,3),dtype="float32"); g.tensors["B"]=TensorSpec(name="B",shape=(1,),dtype="float32")
    g.ops=[Op("Relu","relu",["x"],["r"],{}),Op("Conv","conv",["x","W","B"],["c"],{"strides":[1,1],"pads":[1,1,1,1]}),Op("Add","add",["r","c"],["out"],{})]
    for n in ["r","c","out"]: g.tensors[n]=TensorSpec(name=n,shape=(1,1,4,4),dtype="float32")
    alloc=build_hls_buffer_allocation(g,raw_cfg=_cfg())
    src=emit_dag_top_cpp(g,top_name="deeplearn",weights_mode="embedded",raw_cfg=_cfg(),buffer_allocation=alloc)
    assert "conv2d<" in src
    assert "reinterpret_cast<const op1_wgt_t*>(W0), B0" in src


def test_dag_grouped_conv_codegen_uses_group_aware_kernel():
    g=Graph(name="grouped_conv"); g.inputs=["x"]; g.outputs=["c"]
    g.tensors["x"]=TensorSpec(name="x",shape=(1,4,4,4),dtype="float32")
    g.constants["W"]=np.ones((4,1,3,3),dtype=np.float32); g.constants["B"]=np.zeros((4,),dtype=np.float32)
    g.tensors["W"]=TensorSpec(name="W",shape=(4,1,3,3),dtype="float32"); g.tensors["B"]=TensorSpec(name="B",shape=(4,),dtype="float32")
    g.ops=[Op("Conv","dw",["x","W","B"],["c"],{"strides":[1,1],"pads":[1,1,1,1],"group":4})]
    g.tensors["c"]=TensorSpec(name="c",shape=(1,4,4,4),dtype="float32")
    src=emit_dag_top_cpp(g,top_name="deeplearn",weights_mode="embedded",raw_cfg=_cfg())
    assert "conv2d_grouped<4, 4, 4, 4, 4, 4, 3, 1, 1, 4" in src
    assert "reinterpret_cast<const op0_wgt_t*>(W0), B0" in src
