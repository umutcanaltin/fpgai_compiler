from fpgai.ir.graph import Graph

def emit_model_inst_cpp(graph: Graph) -> str:
    # You can later generate specialized calls here if you want.
    # For now: keep it as a compilation unit that includes the templates.
    return r'''
#include "layers/dense.h"
#include "layers/activations.h"
'''
