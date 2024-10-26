def read_input_stream(model):
    if(model.mode =="inference"):
      read_input_stream_string = "for (int idx = 0; idx < "+str(model.number_of_input_nodes)+"; idx++){"
      read_input_stream_string += """
          #pragma HLS PIPELINE
          floatSdCh input_data = inStream.read();\n""" + model.obj_arch_rep[0].name_of_layer + "_input[idx] = static_cast<float>(input_data.data.to_int())/10000000;\n}\n"
    if(model.mode =="training"):
      read_input_stream_string = "for (int idx = 0; idx < "+str(model.number_of_input_nodes)+"; idx++){"
      read_input_stream_string += """
          #pragma HLS PIPELINE
          floatSdCh input_data = inStream.read();\n""" + model.obj_arch_rep[0].name_of_layer + "_input[idx] = static_cast<float>(input_data.data.to_int())/10000000;\n}\n"
      
      read_input_stream_string += "for (int idx = 0; idx < "+str(model.number_of_output_nodes)+"; idx++){"
      read_input_stream_string += """
          #pragma HLS PIPELINE
          floatSdCh input_data = inStream.read();\n""" +"target_output[idx] = static_cast<float>(input_data.data.to_int())/10000000;\n}\n"
      read_input_stream_string += "floatSdCh input_data = inStream.read();\nint export_weihts = input_data.data.to_int();\n"
    return read_input_stream_string


