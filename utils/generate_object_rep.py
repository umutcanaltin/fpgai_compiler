from architectures.convolution_layer import ConvolutionLayer
from architectures.dense_layer import DenseLayer
from implementations.implementations import dense_layer_imp, conv_layer_imp
def generate_obj_rep(model):
    first_layer= True
    last_layer = False

    
    for i in range(len(model.layers)):
        if(model.layers[i][0]):
            if(i!=0):
                first_layer = False
         
            layer_weights = model.weights[i]
            layer_bias = model.weights[i+1]
            #skip the activation layer
            if(model.layers[i][1] == "conv"):
                if(int(len(model.layers)/2) == i):
                    last_layer = True    
                layer_kernel_shape = model.layers[i][4]
                layer_stride = model.layers[i][5]
                if(first_layer):
                    layer_input_shape = model.first_layer_shape
                else:
                    layer_input_shape = model.obj_arch_rep[int(i/2)-1].output_shape
                new_layer = ConvolutionLayer(mode=model.mode,precision=model.precision,ai_model=model,kernel_shape=layer_kernel_shape,input_shape=layer_input_shape,stride=layer_stride,activation_function=model.layers[i+1][1], weights=layer_weights,bias=layer_bias, is_first_layer=first_layer, name_of_layer="layer_"+str(int(i/2)),is_last_layer= last_layer,layer_order=int(i/2))
                model.obj_arch_rep.append(new_layer)
                if(model.mode == "inference"):
                    new_implementation = conv_layer_imp(precision=model.precision,name_of_layer=new_layer.name_of_layer).get_forward_hls_function()
                    model.layer_function_implementations += new_implementation
                if(model.mode == "training"):
                    new_implementation = conv_layer_imp(precision=model.precision,name_of_layer=new_layer.name_of_layer).get_forward_hls_function()
                    if(new_layer.is_last_layer):
                        new_implementation += conv_layer_imp(precision=model.precision,name_of_layer=new_layer.name_of_layer).get_last_delta_calc_hls_function()
                        new_implementation += model.loss_function
                    else:
                        new_implementation += conv_layer_imp(precision=model.precision,name_of_layer=new_layer.name_of_layer).get_delta_calc_hls_function()

                    new_implementation += conv_layer_imp(precision=model.precision,name_of_layer=new_layer.name_of_layer).get_update_weights_hls_function()
                    model.layer_function_implementations += new_implementation
               


                #model.obj_arch_rep.append(ConvolutionLayer(weights=layer_weights,bias=layer_bias,is_first_layer=first_layer, activation_function=model.layers[i+1][1]))
            elif(model.layers[i][1]== "dense"):
                if(int(len(model.layers)/2) == i):
                    last_layer = True    
                new_layer = DenseLayer(precision=model.precision,ai_model=model, activation_function=model.layers[i+1][1], weights=layer_weights,bias=layer_bias, is_first_layer=first_layer, name_of_layer="layer_"+str(int(i/2)),is_last_layer= last_layer,layer_order=int(i/2),mode=model.mode)
                model.obj_arch_rep.append(new_layer)
                if(model.mode == "inference"):
                    new_implementation = dense_layer_imp(precision=model.precision,name_of_layer=new_layer.name_of_layer).get_forward_hls_function()
                    model.layer_function_implementations += new_implementation
               
                if(model.mode == "training"):
                    new_implementation = dense_layer_imp(precision=model.precision,name_of_layer=new_layer.name_of_layer).get_forward_hls_function()
                    if(new_layer.is_last_layer):
                        new_implementation += dense_layer_imp(precision=model.precision,name_of_layer=new_layer.name_of_layer).get_last_delta_calc_hls_function()
                        new_implementation += model.loss_function
                    else:
                        new_implementation += dense_layer_imp(precision=model.precision,name_of_layer=new_layer.name_of_layer).get_delta_calc_hls_function()

                    new_implementation += dense_layer_imp(precision=model.precision,name_of_layer=new_layer.name_of_layer).get_update_weights_hls_function()
                    model.layer_function_implementations += new_implementation
               
            else:
                raise Exception
        else:
            pass
    return 0