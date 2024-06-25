from __future__ import print_function
from fpgai_engine import fpgai_engine
import argparse

parser = argparse.ArgumentParser(description='fpgai ONNX to SoC Engine')
parser.add_argument('--mode', type=str, default='inference',
                    help='ONNX file name with path')
parser.add_argument('--learning-rate', type=float, default=0.1,
                    help='ONNX file name with path')
parser.add_argument('--learning-rate-type', type=str, default='fixed',
                    help='ONNX file name with path')

parser.add_argument('--training-mode', type=str, default='0',
                    help='ONNX file name with path')

parser.add_argument('--onnx-file-name', type=str, default='my_image_classifier.onnx',
                    help='ONNX file name with path')
parser.add_argument('--precision', type=str, default="float",help='ONNX file name with path')
parser.add_argument('--vitis-hls-location', type=str, default='',help='location of vitis HLS files')
parser.add_argument('--hls-project-name', type=str, default="new",help='usage of existing hls project')
parser.add_argument('--hls-solution-name', type=str, default="float",help='usage of existing hls project solution')
parser.add_argument('--memory-option-weights', type=str, default="float",help='usage of existing hls project solution')
parser.add_argument('--ddr-usage', type=bool, default="float",help='usage of existing hls project solution')
parser.add_argument('--dma-usage', type=bool, default="float",help='usage of existing hls project solution')
parser.add_argument('--hardware-optimization', type=bool, default="float",help='usage of existing hls project solution')
parser.add_argument('--hardware-optimization-option', type=str, default="float",help='usage of existing hls project solution')

#parser.add_argument('--hls-main-func-name', type=str, default="main.cpp",help='usage of existing hls project solution')

args = parser.parse_args()

if __name__ == "__main__":
    _onnx_file_name = args.onnx_file_name
    _precision = args.precision
    _vitis_hls_location = args.vitis_hls_location
    _hls_project_name = args.hls_project_name
    _hls_solution_name = args.hls_solution_name
   
    

    ai_model = fpgai_engine(onnx_file_name=_onnx_file_name, precision = _precision,vitis_hls_location=_vitis_hls_location, hls_project_name= _hls_project_name,
                        hls_solution_name= _hls_solution_name)


    

    


  
