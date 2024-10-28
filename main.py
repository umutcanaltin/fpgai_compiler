from __future__ import print_function
from fpgai_engine import fpgai_engine
import argparse
import numpy as np
parser = argparse.ArgumentParser(description='fpgai ONNX to SoC Engine')
parser.add_argument('--mode', type=str, default='training',
                    help='select engine mode for file generation!')
parser.add_argument('--learning-rate', type=float, default=0.1,
                    help='Learning rate parameter for training!')
parser.add_argument('--onnx-file-name', type=str, default='mlp.onnx',
                    help='ONNX file name with path')


args = parser.parse_args()

if __name__ == "__main__":
    _onnx_file_name = args.onnx_file_name
    _precision = "float"
    _mode = args.mode

    input_shape = (1, 8)
    increment = 0.1
    total_elements = np.prod(input_shape)
    input_data = np.arange(0.1, total_elements * increment + 0.1, increment, dtype=np.float32)
    input_data = input_data.reshape(1, 1, 1,8)
    output_shape = (1, 10)

    ai_model = fpgai_engine(onnx_file_name=_onnx_file_name, precision = _precision,mode=_mode,input_data=input_data,first_layer_shape=[1,8],output_shape=output_shape)


    

    


  
