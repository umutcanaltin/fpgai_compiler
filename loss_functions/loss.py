def MeanSquaredError():
    loss_string = """
float calculate_loss(float *layer_output, float *target_output, float *error_buffer, int output_size) {
    float loss = 0.0f;
    for (int i = 0; i < output_size; i++) {
        float error = target_output[i] - layer_output[i];
        error_buffer[i] = error; 
        loss += error * error;
    }
    return loss / output_size; 
}

"""
    return loss_string



