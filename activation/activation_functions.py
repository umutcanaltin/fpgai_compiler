class Activation_functions():
    def __init__ (self, activation_function="linear", precision ="int", name_of_layer= None):
        self.activation_function = activation_function
        self.precision = precision
        self.name_of_layer = name_of_layer
        self.used_activation_funcs=[]
    def get_activation_function(self,mode):
        activation_string = ""
        if(self.activation_function == "linear"):
            self.used_activation_funcs.append("linear")
            if(mode == "inference"):
                activation_string = self.precision + " activate_"+self.name_of_layer+ "( "+self.precision +" x) { return x; } \n"
            if(mode == "training"):
                activation_string = self.precision + " activate_"+self.name_of_layer+ "( "+self.precision +" x) { return x; } \n"+self.precision + " dactivate_"+self.name_of_layer+"( "+self.precision +" x) { return 1; }\n"

        if(self.activation_function == "relu"):
            self.used_activation_funcs.append("relu")
            if(mode == "inference"):
                activation_string = self.precision + " activate_"+self.name_of_layer+ "( "+self.precision +" x) { return x > 0 ? x : 0; } \n"
            if(mode == "training"):
                activation_string = self.precision + " activate_"+self.name_of_layer+ "( "+self.precision +" x) { return x; } \n"+self.precision +" dactivate_"+self.name_of_layer+"( "+self.precision +" x) { return x > 0 ? 1.0 : 0.0; }\n"

        return activation_string
