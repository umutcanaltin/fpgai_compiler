from architectures.dense_layer import DenseLayer
def get_number_of_input_nodes(model):
    if(isinstance(model.obj_arch_rep[0], DenseLayer)):
        return model.obj_arch_rep[0].weights.shape[1]

def get_number_of_output_nodes(model):
    if(isinstance(model.obj_arch_rep[-1], DenseLayer)):
        return model.obj_arch_rep[-1].weights.shape[0]