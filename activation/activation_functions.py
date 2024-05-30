class Activation_functions():
    def __init__ (self, activation_function="linear", precision ="int"):
        self.activation_function = activation_function
        self.precision = precision
        self.used_activation_funcs=[]
    
    def get_activation_function(self):
        activation_string = ""
        if(self.activation_function == "linear"):
            self.used_activation_funcs.append("linear")
            activation_string = """
int activate( int x) { return x; }
int dactivate( int x) { return 1; }
"""

        if(self.activation_function == "relu"):
            self.used_activation_funcs.append("relu")
            activation_string = """
int activate( int x) { return x; }
int dactivate( int x) { return 1; }
"""


        
        return activation_string
