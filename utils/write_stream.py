    
def write_output_stream(model):
	write_output_stream_string  =  """for (int idx = 0; idx < number_of_outputs; idx++) {
			#pragma HLS PIPELINE
			intSdCh output_data;
			output_data.data = static_cast<ap_int<32>>(""" +model.obj_arch_rep[-1].name_of_layer +"""_output[idx]*10000000); // Convert float to integer
			output_data.keep = -1; // Keep all bytes
			output_data.strb = -1; // All strobes valid
			output_data.user = 0;
			output_data.last = (idx == number_of_outputs - 1) ? 1 : 0;
			output_data.id = 0;
			output_data.dest = 0;
			outStream.write(output_data);
		}}"""
	return write_output_stream_string