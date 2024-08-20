def get_pragmas(model):
    memory_decleration = False
    pragma_string = ""
    if(model.mode == "inference"):
        pragma_string = """
#pragma HLS INTERFACE axis port=outStream \n
#pragma HLS INTERFACE axis port=inStream  \n
"""
    pragma_string += "#pragma HLS INTERFACE ap_ctrl_none port=return \n"
    if(memory_decleration):
        for i in range(len(model.obj_arch_rep)):
            pragma_string += "#pragma HLS INTERFACE bram port =  MEM_" + str(i) + "\n"

    if(model.mode == "training"):
        pragma_string = """
#pragma HLS INTERFACE axis port=outStream \n
#pragma HLS INTERFACE axis port=inStream  \n
"""
    pragma_string += "#pragma HLS INTERFACE ap_ctrl_none port=return \n"
    if(memory_decleration):
        for i in range(len(model.obj_arch_rep)):
            pragma_string += "#pragma HLS INTERFACE bram port =  MEM_" + str(i) + "\n"
    return pragma_string