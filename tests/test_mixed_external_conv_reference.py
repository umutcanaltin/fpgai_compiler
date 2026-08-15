from __future__ import annotations

import numpy as np

from fpgai.ir.graph import Graph, Op
from fpgai.validation.mixed_external_hls import execute_mixed_graph_reference


class _NoExternal:
    def reference_for(self, _operator_id):
        return None


def test_conv_reference_uses_graph_constants_and_nchw_shape():
    g=Graph('conv_ref')
    g.inputs=['input']; g.outputs=['output']
    g.add_tensor('input',(1,1,2,2)); g.add_tensor('conv',(1,1,2,2)); g.add_tensor('output',(1,1,2,2))
    w=np.zeros((1,1,3,3),dtype=np.float32); w[0,0,1,1]=2.0
    g.constants={'w':w,'b':np.array([1.0],dtype=np.float32)}
    g.ops=[
        Op('Conv','conv0',['input','w','b'],['conv'],{'pads':[1,1,1,1],'strides':[1,1]}),
        Op('Relu','relu0',['conv'],['output'],{}),
    ]
    out=execute_mixed_graph_reference(g,_NoExternal(),np.array([-1,0,1,2],dtype=np.float32))
    np.testing.assert_allclose(out,[0,1,3,5],rtol=0,atol=1e-6)
