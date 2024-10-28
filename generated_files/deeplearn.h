
// deeplearn.h

        #ifndef DEEPLEARN_H
        #define DEEPLEARN_H
        #include <hls_stream.h>
        #include <ap_axi_sdata.h>
        #include <ap_int.h>
        typedef ap_axis<32, 2, 5, 6> floatSdCh;
        int export_weihts;

        #define number_of_inputs 8
#define number_of_outputs 4
const float learning_rate = 0.1;
void deeplearn(hls::stream<floatSdCh> &inStream, hls::stream<floatSdCh> &outStream);
#endif 
