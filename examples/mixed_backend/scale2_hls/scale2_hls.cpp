#include <ap_int.h>

extern "C" void scale2_hls(
    ap_int<16> input_data,
    ap_uint<1> input_valid,
    ap_int<16>& output_data,
    ap_uint<1>& output_valid) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE ap_none port=input_data
#pragma HLS INTERFACE ap_none port=input_valid
#pragma HLS INTERFACE ap_none port=output_data
#pragma HLS INTERFACE ap_none port=output_valid
#pragma HLS PIPELINE II=1
    output_data = input_data * 2;
    output_valid = input_valid;
}
