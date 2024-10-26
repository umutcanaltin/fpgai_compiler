from google.protobuf.json_format import MessageToDict
import onnx
from onnx import numpy_helper
import numpy as np

def quantize_weights(weights_list, scale_factor):
    quantized_weights = []
    for weight in weights_list:
        # Scale and quantize weights
        scaled_weights = weight * scale_factor
        quantized_weights.append(np.round(scaled_weights).astype(np.int8))  # Convert to int8
    
    return quantized_weights

def get_model_weights(model, quantization = False):
    weight_list= []
    for weights in model.graph.initializer:
        wg = numpy_helper.to_array(weights)
        weight_list.append(wg)
        print(wg.shape)
    if(quantization):
        weight_list = quantize_weights(weights_list=weight_list, scale_factor=127)
    return weight_list

def get_model_arch(model):
    layer_list= []

    for layers in model.graph.node:
        not_a_layer = False
        dictionary_layer = MessageToDict(layers)


        if(not (dictionary_layer["output"][0][:3] == "/co" or dictionary_layer["output"][0][:3]=="/fc" or dictionary_layer["output"][0][:2] == "co" or dictionary_layer["output"][0][:2]=="fc" or dictionary_layer["name"][:3] == "/co" or dictionary_layer["name"][:3]=="/fc" or dictionary_layer["name"][:2] == "co" or dictionary_layer["name"][:2]=="fc")):
            not_a_layer= True
          
        if(not not_a_layer):
            layer_stride = []
            layer_kernel_shape =[]
            layer_type = "dense"
            
            
            if(dictionary_layer["output"][0][:3] == "/co" or dictionary_layer["output"][0][:2] == "co"):
                layer_type = "conv"
                layer_attribute = dictionary_layer["attribute"]
                for k in range(len(layer_attribute)):
                    if(layer_attribute[k]["name"] == "strides"):
                        layer_stride = layer_attribute[k]['ints']
                    if(layer_attribute[k]["name"] == "kernel_shape"):
                        layer_kernel_shape = layer_attribute[k]['ints']
               
            layer_bias = dictionary_layer["input"][2]
            if(dictionary_layer["input"][1][:3]=="/co" or dictionary_layer["input"][1][:3]=="/fc" or dictionary_layer["input"][1][:2]=="co" or dictionary_layer["input"][1][:2]=="fc"  ):
                layer_weight = dictionary_layer["input"][1]
            else:
                layer_weight = layer_bias[:-4]+"weight"

            layer_info = [not not_a_layer,layer_type, layer_weight, layer_bias,layer_kernel_shape,layer_stride]
            layer_list.append(layer_info)
        if(not_a_layer):
            if(dictionary_layer["input"][0][:3] == "/co" or dictionary_layer["input"][0][:3]=="/fc" or dictionary_layer["input"][0][:2] == "co" or dictionary_layer["input"][0][:2]=="fc"):
                    
                try:
                    if(dictionary_layer["output"][0][:5]=="/Relu"):
                        activation_func = "relu"
                        layer_info = [not not_a_layer, activation_func]
                        layer_list.append(layer_info)
                    if(dictionary_layer["output"][0][:4]=="relu"):
                        activation_func = "relu"
                        layer_info = [not not_a_layer, activation_func]
                        layer_list.append(layer_info)
                    if(dictionary_layer["output"][0][:3]=="sig"):
                        activation_func = "sigmoid"
                        layer_info = [not not_a_layer, activation_func]
                        layer_list.append(layer_info)
                         
                except:
                    raise Exception('Our tool does not support this activation function!')

    
    return layer_list

def verify_model(model):
    try:
        onnx.checker.check_model(model)
        return 1
    except onnx.checker.ValidationError as e:
        return(f"The model is invalid: {e}")
