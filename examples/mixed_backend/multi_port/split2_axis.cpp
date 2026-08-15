#include <ap_int.h>
#include <hls_stream.h>

void split2_axis(
    hls::stream<ap_int<16>>& input,
    hls::stream<ap_int<16>>& left,
    hls::stream<ap_int<16>>& right) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=left
#pragma HLS INTERFACE axis port=right
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1
    ap_int<16> value = input.read();
    left.write(value);
    right.write(value);
}
