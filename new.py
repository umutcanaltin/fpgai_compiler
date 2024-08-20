import onnx
import numpy as np

# Function to parse convolution layer
def parse_conv_layer(node, input_shape):
    print(f"Parsing Conv Layer: {node.name}")
    # Example: Extract weights for convolution from the node or use placeholders
    filters = [0.1] * (3 * 3)  # Replace with actual filter extraction logic
    filter_size = 3  # Example: Assuming 3x3 filters
    num_filters = 1  # Replace with actual logic to determine the number of filters
    return {
        'type': 'conv',
        'filter_size': filter_size,
        'num_filters': num_filters,
        'weights': filters,
        'input_shape': input_shape,
        'output_shape': (input_shape[0] - filter_size + 1, input_shape[1] - filter_size + 1)
    }

# Function to parse dense layer
def parse_dense_layer(node, input_size):
    print(f"Parsing Dense Layer: {node.name}")
    # Example: Extract weights for dense layer from the node or use placeholders
    weights = [0.1] * input_size  # Replace with actual weight extraction logic
    num_units = 10  # Example: Assuming 10 output units
    return {
        'type': 'dense',
        'input_size': input_size,
        'num_units': num_units,
        'weights': weights
    }

# Function to generate C++ HLS code
def generate_hls_code(layers):
    hls_code = ""

    # Convolution layer template
    conv_template = """
    static void compute_convolution(const float* in, float* conv_out, int input_size, int filter_size, int output_size, int num_filters) {{
        const float filters[{num_filters} * {filter_size} * {filter_size}] = {{{weights}}};

        for (int f = 0; f < num_filters; f++) {{
            for (int x = 0; x < output_size; x++) {{
                for (int y = 0; y < output_size; y++) {{
                    float conv_sum = 0.0;
                    for (int fx = 0; fx < filter_size; fx++) {{
                        for (int fy = 0; fy < filter_size; fy++) {{
                            int input_x = x + fx;
                            int input_y = y + fy;
                            int input_index = input_x * input_size + input_y;
                            int filter_index = f * (filter_size * filter_size) + (fx * filter_size + fy);
                            conv_sum += in[input_index] * filters[filter_index];
                        }}
                    }}
                    conv_out[f * output_size * output_size + x * output_size + y] = std::max(0.0f, conv_sum);
                }}
            }}
        }}
    }}
    """

    # Dense layer template
    dense_template = """
    static void dense_layer(const float* conv_out, float* dense_out, int input_size, int num_units) {{
        const float weights[{num_units} * {input_size}] = {{{weights}}};
        const float biases[{num_units}] = {{{biases}}};

        for (int unit = 0; unit < num_units; unit++) {{
            float sum = biases[unit];
            for (int i = 0; i < input_size; i++) {{
                int weight_index = unit * input_size + i;
                sum += conv_out[i] * weights[weight_index];
            }}
            dense_out[unit] = std::max(0.0f, sum);
        }}
    }}
    """

    # Generate code for each layer
    for layer in layers:
        if layer['type'] == 'conv':
            hls_code += conv_template.format(
                num_filters=layer['num_filters'],
                filter_size=layer['filter_size'],
                weights=", ".join(map(str, layer['weights']))
            )
        elif layer['type'] == 'dense':
            hls_code += dense_template.format(
                num_units=layer['num_units'],
                input_size=layer['input_size'],
                weights=", ".join(map(str, layer['weights'])),
                biases=", ".join(["1.0"] * layer['num_units'])  # Placeholder biases
            )

    # Ensure at least one conv and dense layer is present
    if not any(layer['type'] == 'conv' for layer in layers):
        raise ValueError("No convolution layer found in the ONNX model.")
    
    if not any(layer['type'] == 'dense' for layer in layers):
        raise ValueError("No dense layer found in the ONNX model.")

    # Main function template
    main_template = """
    extern "C" {{
    void cnn_infer(float* in, float* out) {{
        static float conv_out[{conv_out_size}];
        compute_convolution(in, conv_out, {input_size}, {filter_size}, {output_size}, {num_filters});
        dense_layer(conv_out, out, {dense_input_size}, {num_units});
    }}
    }}
    """
    
    # Assuming one conv and one dense layer for simplicity
    conv_layer = next(l for l in layers if l['type'] == 'conv')
    dense_layer = next(l for l in layers if l['type'] == 'dense')

    hls_code += main_template.format(
        conv_out_size=conv_layer['num_filters'] * conv_layer['output_shape'][0] * conv_layer['output_shape'][1],
        input_size=conv_layer['input_shape'][0],
        filter_size=conv_layer['filter_size'],
        output_size=conv_layer['output_shape'][0],
        num_filters=conv_layer['num_filters'],
        dense_input_size=conv_layer['num_filters'] * conv_layer['output_shape'][0] * conv_layer['output_shape'][1],
        num_units=dense_layer['num_units']
    )

    return hls_code

# Function to parse ONNX model
def parse_onnx_model(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    layers = []
    input_shape = (28, 28)  # Placeholder for input shape

    print("Parsing ONNX model layers...")
    for node in graph.node:
        print(f"Found layer: {node.op_type} with name {node.name}")
        if 'Conv2d' in node.name:  # Adjusted to detect Conv2d layers
            conv_layer = parse_conv_layer(node, input_shape)
            layers.append(conv_layer)
            input_shape = conv_layer['output_shape']
        elif 'Linear' in node.name:  # Adjusted to detect Linear (dense) layers
            dense_layer = parse_dense_layer(node, np.prod(input_shape))
            layers.append(dense_layer)
    
    return layers

# Main function to generate HLS code from ONNX model
def generate_hls_from_onnx(onnx_file, output_cpp_file):
    layers = parse_onnx_model(onnx_file)
    if not layers:
        raise ValueError("No layers were parsed from the ONNX model.")
    
    hls_code = generate_hls_code(layers)
    
    # Write the generated HLS code to a file
    with open(output_cpp_file, 'w') as f:
        f.write(hls_code)



onnx_file = 'my_image_classifier.onnx'
output_cpp_file = 'cnn_infer.cpp'
generate_hls_from_onnx(onnx_file, output_cpp_file)
