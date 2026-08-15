#include <ap_int.h>
#include <hls_stream.h>

void add2_axis(
    hls::stream<ap_int<16>>& left_done,
    hls::stream<ap_int<16>>& right_done,
    hls::stream<ap_int<16>>& output) {
#pragma HLS INTERFACE axis port=left_done
#pragma HLS INTERFACE axis port=right_done
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1
    ap_int<16> left_value = left_done.read();
    ap_int<16> right_value = right_done.read();
    output.write(left_value + right_value);
}
