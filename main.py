from __future__ import print_function
from ai_model import AI_Model
import argparse

parser = argparse.ArgumentParser(description='fpgai ONNX to SoC Engine')
parser.add_argument('--onnx-file-name', type=str, default='my_image_classifier.onnx',
                    help='ONNX file name with path')
parser.add_argument('--precision', type=str, default="float64",help='ONNX file name with path')
parser.add_argument('--vitis-hls-location', type=str, default='',help='location of vitis HLS files')
parser.add_argument('--hls-project-name', type=str, default="new",help='usage of existing hls project')
parser.add_argument('--hls-solution-name', type=str, default="float64",help='usage of existing hls project solution')
parser.add_argument('--memory-option-weights', type=str, default="float64",help='usage of existing hls project solution')
parser.add_argument('--ddr-usage', type=bool, default="float64",help='usage of existing hls project solution')
parser.add_argument('--dma-usage', type=bool, default="float64",help='usage of existing hls project solution')
parser.add_argument('--hardware-optimization', type=bool, default="float64",help='usage of existing hls project solution')
parser.add_argument('--hardware-optimization-option', type=str, default="float64",help='usage of existing hls project solution')

#parser.add_argument('--hls-main-func-name', type=str, default="main.cpp",help='usage of existing hls project solution')

args = parser.parse_args()

if __name__ == "__main__":
    _onnx_file_name = args.onnx_file_name
    _ui = args.ui
    _precision = args.precision
    _vitis_hls_location = args.vitis_hls_location
    _hls_project_name = args.hls_project_name
    _hls_solution_name = args.hls_solution_name
    _memory_option_weights = args.memory_option_weights
    _ddr_usage = args.ddr_usage
    _dma_usage = args.dma_usage
    _hardware_optimization = args.hardware_optimization
    hardware_optimization_option = args.hardware_optimization_option

    ai_model = AI_Model(onnx_file_name=_onnx_file_name, precision = _precision,vitis_hls_location=_vitis_hls_location, hls_project_name= _hls_project_name,
                        hls_solution_name= _hls_solution_name,memory_option_weights=_memory_option_weights , use_DMA=_dma_usage,user_DDR=_ddr_usage)


    

    


  
