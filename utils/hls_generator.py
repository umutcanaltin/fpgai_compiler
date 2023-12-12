from keras.models import model_from_json
import argparse
from readers import H5_reader, Json_reader
import numpy as np
import os
from cpp_details import Details
from run_c_test import Tester
from writers import Writers
parser = argparse.ArgumentParser()
parser.add_argument("--json_file", default = "model.json" , type = str , help = "set the models json file for architecture of network,  default : model.json ")
parser.add_argument("--h5_file", default= "model.h5", type = str , help = "set the models h5 file for weights of network,  default : model.h5")
parser.add_argument("--precision", default= 30, type = int, help = "set the float precision for weights of network,  default : 30 ,   This may cause problem in network output so be carefull when choosing the value")
parser.add_argument("--output_file", default = "main.c" , type = str, help = "set the output c/c++ file for test purposes " )
parser.add_argument("--output_folder_name", default = "output_cpp_test" , type = str, help = "Set the output files folder name configuration for main test cpp file" )
parser.add_argument("--number_test_input", default = 20 , type = int )

args = parser.parse_args()
json_file = args.json_file
h5_file = args.h5_file
precision = args.precision
output_file = args.output_file
output_folder_name = args.output_folder_name
number_of_test_input = args.number_test_input
a= os.path.dirname(os.path.realpath(__file__))
if( not os.path.isdir(output_folder_name)):
    os.mkdir(output_folder_name)


script_dir = os.path.dirname(__file__)
rel_path = output_folder_name + "/" + output_file
abs_file_path = os.path.join(script_dir, rel_path)
print(abs_file_path)
f= open(abs_file_path,"w")
rel_path_cpp = output_folder_name + "/" + "main.cpp"
abs_file_path_cpp = os.path.join(script_dir, rel_path_cpp)
f_cpp = open(abs_file_path_cpp,"w")
    
details = Details()
h5_reader = H5_reader()
writer = Writers(script_dir,output_folder_name)
json_reader = Json_reader()


f.write(details.set_main_libs())
f.write(details.set_main_initializer())
f_cpp.write(details.set_main_libs_cpp())
f_cpp.write(details.set_main_initializer_cpp())

ordered_layers = json_reader.get_layers()
print("Ordered layer list is ready.")


test_vector = details.generate_input_vector(json_reader.get_input_shape(ordered_layers[0]))
test_vector_string = details.write_input_vector(test_vector[0],precision,0)
f.write(test_vector_string)



for layer in range(len(ordered_layers)):

    layer_name = ordered_layers[layer]
    #details.set_initial_arrays(layer)

    if (layer_name[:3]=="con"):
        kernel_size,filter_value, output_dim, initial_arrays, input_channel = details.convolution_settings(ordered_layers, layer)
        filter_string =details.filter_string(kernel_size,filter_value,ordered_layers,layer,precision,input_channel)
        f.write(filter_string)
        f.write(initial_arrays)
        f_cpp.write(filter_string)
        f_cpp.write(initial_arrays)
        writer.write_convolution_function(layer_name,json_reader.get_input_shape(layer_name)[1:] , json_reader.get_kernel_size(layer_name),json_reader.get_filter_value(layer_name))

    elif(layer_name[:3]=="max"):
        f.write(details.max_pooling_settings(ordered_layers, layer))
        f_cpp.write(details.max_pooling_settings(ordered_layers, layer))

    elif(layer_name[:3]=="den"):
        f.write(details.set_dense_weights(ordered_layers[layer],precision))
        f.write(details.set_dense_bias(layer_name))
        f.write(details.dense_settings(ordered_layers, layer))
        f_cpp.write(details.set_dense_weights(ordered_layers[layer],precision))
        f_cpp.write(details.set_dense_bias(layer_name))
        f_cpp.write(details.dense_settings(ordered_layers, layer))
    else:
        if(layer_name[:3] != "fla"):
            raise Exception('Not Supported Layer!!!')

f.write(details.set_default_end(ordered_layers))
f_cpp.write(details.set_end_cpp())
# Model reconstruction from JSON file
model = None
with open(json_file, 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(h5_file)
#predict
model_test_prediction = model.predict(test_vector)



f.close()
f_cpp.close()

tester = Tester()
tester.run_test_c(model_test_prediction[0])

script_dir = os.path.dirname(__file__)
rel_path =  "test.cpp"
abs_file_path = os.path.join(a, rel_path)


rel_path_1 =  "usps.h5"
abs_file_path_1 = os.path.join(a, rel_path_1)

from test_hls import Test_HLS
test_hls = Test_HLS(precision,abs_file_path,abs_file_path_1)
test_hls.write_start()
test_hls.write_network_input()
test_hls.write_end()
hls_test_input = test_hls.write()
#print( model.predict(np.array([hls_test_input])))