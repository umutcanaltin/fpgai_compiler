from __future__ import print_function
from fpgai_engine import fpgai_engine
import argparse

parser = argparse.ArgumentParser(description='fpgai ONNX to SoC Engine')
parser.add_argument('--mode', type=str, default='training',
                    help='ONNX file name with path')
parser.add_argument('--learning-rate', type=float, default=0.1,
                    help='ONNX file name with path')
parser.add_argument('--onnx-file-name', type=str, default='image_classifier_2_exported.onnx',
                    help='ONNX file name with path')
parser.add_argument('--precision', type=str, default="float",help='ONNX file name with path')


args = parser.parse_args()

if __name__ == "__main__":
    _onnx_file_name = args.onnx_file_name
    _precision = args.precision
    _mode = args.mode
    ai_model = fpgai_engine(onnx_file_name=_onnx_file_name, precision = _precision,mode=_mode)


    

    


  
