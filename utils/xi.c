```

#include "ap_axi_sdata.h"
#include "hls_stream.h"

 

typedef ap_axiu<32, 1, 1, 1> pkt;

#define in_N 784
#define out_N 10
 

void deeplearn(hls::stream<pkt> s_in0[in_N], hls::stream<pkt> s_out[out_N]) {

#pragma HLS INTERFACE axis port=s_in0
#pragma HLS INTERFACE axis port=s_out

 

pkt data_in[in_N];
pkt data_out[out_N];

 

    for (unsigned i = 0; i < in_N; i++) {

    data_in0[i] = s_in0[i].read();


    }


    for(unsigned k = 0; k < out_N; k++){

    data_out[k].data =1;

    }

 

data_out[i].dest = data_in1[i].dest;

data_out[i].id = data_in1[i].id;

data_out[i].keep = data_in1[i].keep;

data_out[i].last = data_in1[i].last;

data_out[i].strb = data_in1[i].strb;

data_out[i].user = data_in1[i].user;

 

s_out[i].write(data_out[i]);

 

    if(data_out[i].last){

    break;

    }





