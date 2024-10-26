from architectures.convolution_layer import ConvolutionLayer
from architectures.dense_layer import DenseLayer

def construct_main_function(model):
    memory_decleration = False
    main_func_string = ""
    main_func_string += "void " + model.main_func_name + "( hls::stream<floatSdCh> &inStream, "
    
    if(memory_decleration):
        for i in range(len(model.obj_arch_rep)):
            if(isinstance(model.obj_arch_rep[i], ConvolutionLayer)):
                main_func_string += model.precision + " MEM_"+ str(i) +"["+model.obj_arch_rep[i]
                #dolacak

            if(isinstance(model.obj_arch_rep[i], DenseLayer)):
                main_func_string +=model.precision + " MEM_"+ str(i) +"["+str(len(model.obj_arch_rep[i].weights)*len(model.obj_arch_rep[i].weights[0]) + len(model.obj_arch_rep[i].bias)) + "],"
                    
    if(model.mode == "inference"):
        main_func_string += "hls::stream<floatSdCh> &outStream){" + "\n"
    if(model.mode == "training"):
        main_func_string += " int mode_var, "
        main_func_string += "hls::stream<floatSdCh> &outStream){"+ "\n"
        #todo list input eklenecek
    return main_func_string
    