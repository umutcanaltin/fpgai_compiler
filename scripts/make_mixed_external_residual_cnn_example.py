from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def write_model(path: Path) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    w0=np.zeros((1,1,3,3),dtype=np.float32); w0[0,0,1,1]=1.0
    b0=np.zeros((1,),dtype=np.float32)
    w1=np.zeros((1,1,3,3),dtype=np.float32); w1[0,0,1,1]=0.5
    b1=np.zeros((1,),dtype=np.float32)
    init=[numpy_helper.from_array(w0,'w0'),numpy_helper.from_array(b0,'b0'),numpy_helper.from_array(w1,'w1'),numpy_helper.from_array(b1,'b1')]
    nodes=[
      helper.make_node('Conv',['input','w0','b0'],['conv0'],name='conv0',pads=[1,1,1,1],strides=[1,1]),
      helper.make_node('Relu',['conv0'],['relu0'],name='relu0'),
      helper.make_node('Conv',['relu0','w1','b1'],['conv1'],name='conv1',pads=[1,1,1,1],strides=[1,1]),
      helper.make_node('ScaleBias',['input'],['skip'],name='scale_bias_skip',domain='community.fpgai',scale=1.0,bias=0.0),
      helper.make_node('Add',['conv1','skip'],['sum'],name='add0'),
      helper.make_node('Relu',['sum'],['output'],name='relu1'),
    ]
    g=helper.make_graph(nodes,'mixed_external_residual_cnn',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,1,4,4])],[helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,1,4,4])],initializer=init)
    model=helper.make_model(g,opset_imports=[helper.make_opsetid('',13),helper.make_opsetid('community.fpgai',1)])
    path.parent.mkdir(parents=True,exist_ok=True); onnx.save(model,path); return path

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='models/mixed_external_residual_cnn.onnx'); args=ap.parse_args(); print(write_model(Path(args.out)))
