    
def write_output_stream(model):
    write_output_stream_string = "for (int idx = 0; idx < "+str(model.number_of_output_nodes)+"; idx++){"
    write_output_stream_string += """


			intSdCh valOut;
			intSdCh valIn;
			valIn = stream_input[idx];
			valIn.data = layer_1_output[idx];
			valOut.data = valIn.data ;

			valOut.keep = valIn.keep;
			valOut.strb = valIn.strb;
			valOut.user = valIn.user;
			valOut.last = valIn.last;
			

"""
    write_output_stream_string += "              if(idx=="+str(model.number_of_output_nodes-1) +"){"
    write_output_stream_string += """
				valOut.last = 1;
			}
			valOut.id = valIn.id;
			valOut.dest = valIn.dest;
			outStream.write(valOut);
}
}
"""
    return write_output_stream_string