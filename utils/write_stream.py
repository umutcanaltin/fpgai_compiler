from architectures.dense_layer import DenseLayer
from architectures.convolution_layer import ConvolutionLayer   
def write_output_stream(model):
	if(model.mode == "inference"):
		write_output_stream_string  =  """for (int idx = 0; idx < number_of_outputs; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(""" +model.obj_arch_rep[-1].name_of_layer +"""_output[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = (idx == number_of_outputs - 1) ? 1 : 0;
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			}}"""
	if(model.mode == "training"):
		write_output_stream_string  =  """
		if(export_weihts == 0){
		for (int idx = 0; idx < number_of_outputs; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(""" +model.obj_arch_rep[-1].name_of_layer +"""_output[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = (idx == number_of_outputs - 1) ? 1 : 0;
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			} } 
			else{
		"""


		for i in range(len(model.obj_arch_rep)):
			write_output_stream_string += """

			for (int idx = 0; idx < """
			if(isinstance(model.obj_arch_rep[i], DenseLayer)):				
				write_output_stream_string += str(model.obj_arch_rep[i].input_shape * model.obj_arch_rep[i].output_shape)
			if(isinstance(model.obj_arch_rep[i], ConvolutionLayer)):
				write_output_stream_string += str(model.obj_arch_rep[i].kernel_shape[0]* model.obj_arch_rep[i].kernel_shape[1])
				
			write_output_stream_string += """; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(""" +model.obj_arch_rep[i].name_of_layer +"""_weights[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = 0;
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			}


			for (int idx = 0; idx < """
			if(isinstance(model.obj_arch_rep[i], DenseLayer)):				
				write_output_stream_string += str(model.obj_arch_rep[i].output_shape)
			if(isinstance(model.obj_arch_rep[i], ConvolutionLayer)):
				write_output_stream_string += str(model.obj_arch_rep[i].bias.shape[0])
			
			write_output_stream_string += """; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(""" +model.obj_arch_rep[i].name_of_layer +"""_bias[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = """
			if(i == len(model.obj_arch_rep)-1):
				write_output_stream_string += "(idx == number_of_outputs - 1) ? 1 : 0;"
			else:
				write_output_stream_string += "0;"
			write_output_stream_string += """
				
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			}

			"""

		write_output_stream_string+= """	} }
			"""
	return write_output_stream_string