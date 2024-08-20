def read_input_stream(model):
    read_input_stream_string = "for (int idx = 0; idx < "+str(model.number_of_input_nodes)+"; idx++){"
    read_input_stream_string += """
			intSdCh valOut;
			intSdCh valIn = inStream.read();
			stream_input[idx] = valIn;
			input_matrix[idx] =valIn.data;
}
"""
    return read_input_stream_string