from __future__ import print_function
from ai_model import AI_Model
import argparse

parser = argparse.ArgumentParser(description='ONNX to SoC Engine')
parser.add_argument('--onnx-file-name', type=str, default='my_image_classifier.onnx',
                    help='ONNX file name with path')
parser.add_argument('--ui', type=bool, default=False,help='ONNX file name with path')
parser.add_argument('--precision', type=str, default="float64",help='ONNX file name with path')
args = parser.parse_args()

if __name__ == "__main__":
    ai_model = AI_Model(onnx_file_name= args.onnx_file_name)
