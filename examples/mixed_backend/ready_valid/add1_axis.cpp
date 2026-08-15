#include <ap_int.h>
#include <hls_stream.h>

void add1_axis(hls::stream<ap_int<16>>& input, hls::stream<ap_int<16>>& output) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1
    ap_int<16> value = input.read();
    output.write(value + 1);
}
