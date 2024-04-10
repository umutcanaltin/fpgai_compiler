from google.protobuf.json_format import MessageToDict
import onnx
from onnx import numpy_helper

def get_model_weights(model):
    weight_list= []
    for weights in model.graph.initializer:
        #print(MessageToDict(weights))
        weight_list.append(numpy_helper.to_array(weights))
    return weight_list

def get_model_arch(model):
    layer_list= []
    is_first_layer = False
    for layers in model.graph.node:
        
        
        not_a_layer = False
        dictionary_layer = MessageToDict(layers)
        print(dictionary_layer)
      
        try:
            if(not (dictionary_layer["output"][0][:2] == "co" or dictionary_layer["output"][0][:2]=="fc")):
                not_a_layer= True
            if(not not_a_layer):
                layer_weight = dictionary_layer["input"][1]
                layer_bias = dictionary_layer["input"][2]
                layer_info = [not_a_layer, layer_weight, layer_bias]
            if(not_a_layer):
                if(dictionary_layer["input"][0][:2] == "co" or dictionary_layer["input"][0][:2]=="fc"):
                    try:
                        if(dictionary_layer["output"][0][:4]=="relu"):
                            activation_func = "relu"
                            layer_info = [not_a_layer, activation_func]
                         
                    except:
                        raise Exception('Our tool does not support this activation function!')
            
            layer_list.append(layer_info)
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
