def get_pragmas(model):
    memory_decleration = False
    pragma_string = ""
    if(model.mode == "inference"):
        pragma_string = """
#pragma HLS INTERFACE axis port=outStream \n
#pragma HLS INTERFACE axis port=inStream  \n
"""
    pragma_string += "#pragma HLS INTERFACE ap_ctrl_none port=return \n"


    if(model.mode == "training"):
        pragma_string = """
#pragma HLS INTERFACE axis port=outStream \n
#pragma HLS INTERFACE axis port=inStream  \n
"""
    pragma_string += "#pragma HLS INTERFACE ap_ctrl_none port=return \n"
    return pragma_string