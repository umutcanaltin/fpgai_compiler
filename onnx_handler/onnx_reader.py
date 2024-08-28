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
        weight_list.append(numpy_helper.to_array(weights))

    if(quantization):
        weight_list = quantize_weights(weights_list=weight_list, scale_factor=127)
    return weight_list

def get_model_arch(model):
    layer_list= []

    for layers in model.graph.node:
        not_a_layer = False
        print(layers)
        dictionary_layer = MessageToDict(layers)
        try:
            if(not (dictionary_layer["output"][0][:2] == "co" or dictionary_layer["output"][0][:2]=="fc")):
                not_a_layer= True
            if(not not_a_layer):
                layer_weight = dictionary_layer["input"][1]
                layer_bias = dictionary_layer["input"][2]
                layer_type = "dense"
                if(dictionary_layer["output"][0][:2] == "co"):
                    layer_type = "conv"
                layer_info = [not not_a_layer,layer_type, layer_weight, layer_bias]
                layer_list.append(layer_info)
            if(not_a_layer):
                if(dictionary_layer["input"][0][:2] == "co" or dictionary_layer["input"][0][:2]=="fc"):
                    
                    try:
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
        except:
            #not in the scope for our tool!
            continue
        
        #layer_list.append(MessageToDict(layers)["input"])
    
    return layer_list

def verify_model(model):
    try:
        onnx.checker.check_model(model)
        return 1
    except onnx.checker.ValidationError as e:
        return(f"The model is invalid: {e}")
