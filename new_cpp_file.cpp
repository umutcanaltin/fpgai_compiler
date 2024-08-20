void conv_forwardlayer_0( int * input, int  * filters, int filter_number, int filter_size_1, int filter_size_2,int input_size_1, int input_size_2, int * output){
    for(int filter_index = 0 ; filter_index < filter_number ;  filter_index++ ){
      
        for(int conv_x_index=0 ; conv_x_index < input_size_1 - filter_size_1 + 1 ; conv_x_index ++ ){ 
            for(int conv_y_index=0 ; conv_y_index < input_size_2 - filter_size_2 + 1 ; conv_y_index ++ ){
           
            float conv_sum = 0;

                for(int kernel_x_index=0 ; kernel_x_index < filter_size_1 ; kernel_x_index ++ ){  
                    for(int kernel_y_index =0; kernel_y_index < filter_size_2 ; kernel_y_index++){ 
                        conv_sum += *(filters + kernel_y_index + filter_size_2*kernel_x_index)* *(input + conv_y_index + kernel_y_index + input_size_2*kernel_x_index);
                        
                    } 
                }
            *(output + conv_y_index  + conv_x_index*input_size_2) = activate_layer_0(conv_sum); 
            }
        }
    } 
}
        void dense_forward_layer_1( int * layer_input, int* layer_bias, int * layer_weights , int * layer_output ,int number_of_input_nodes, int number_of_output_nodes){

   for (int i=0; i<number_of_output_nodes; i++) {

      int activation = *(layer_bias+i);
      for (int j=0; j<number_of_input_nodes; j++) {

         activation += *(layer_weights + number_of_output_nodes*i + j) * (*(layer_input + i));

      }
      *(layer_output + i)= activate_layer_1(activation);

   }

}
        void dense_forward_layer_2( int * layer_input, int* layer_bias, int * layer_weights , int * layer_output ,int number_of_input_nodes, int number_of_output_nodes){

   for (int i=0; i<number_of_output_nodes; i++) {

      int activation = *(layer_bias+i);
      for (int j=0; j<number_of_input_nodes; j++) {

         activation += *(layer_weights + number_of_output_nodes*i + j) * (*(layer_input + i));

      }
      *(layer_output + i)= activate_layer_2(activation);

   }

}
        void dense_forward_layer_3( int * layer_input, int* layer_bias, int * layer_weights , int * layer_output ,int number_of_input_nodes, int number_of_output_nodes){

   for (int i=0; i<number_of_output_nodes; i++) {

      int activation = *(layer_bias+i);
      for (int j=0; j<number_of_input_nodes; j++) {

         activation += *(layer_weights + number_of_output_nodes*i + j) * (*(layer_input + i));

      }
      *(layer_output + i)= activate_layer_3(activation);

   }

}
        
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define number_of_inputs None
#define number_of_outputs 10
const float lr = 0.1;
int layer_0_input[1];
int layer_0_output[1];
int layer_0_weights[1] = {[[-15  16]
 [ -2  49]]};
int layer_0_bias[1] = {16};
int activate_layer_0( int x) { return x; } 
int layer_1_input[16];
int layer_1_output[12];
int layer_1_weights[192] = {-2 ,13 ,-3 ,9 ,-18 ,29 ,12 ,-11 ,18 ,17 ,-13 ,22 ,-22 ,1 ,28 ,19 ,2 ,8 ,-18 ,-11 ,-17 ,3 ,-24 ,-17 ,28 ,25 ,11 ,16 ,-21 ,-16 ,-7 ,22 ,-2 ,23 ,-18 ,-6 ,-21 ,3 ,-9 ,-7 ,-17 ,22 ,-20 ,-27 ,31 ,22 ,24 ,12 ,-28 ,21 ,-22 ,12 ,-29 ,12 ,-2 ,8 ,0 ,-15 ,5 ,-12 ,30 ,23 ,-1 ,24 ,-24 ,-11 ,2 ,21 ,26 ,-16 ,-24 ,-20 ,31 ,-1 ,-21 ,-12 ,-5 ,-29 ,15 ,31 ,-24 ,15 ,13 ,-17 ,15 ,3 ,-28 ,-7 ,-13 ,-19 ,-6 ,6 ,15 ,-19 ,-16 ,22 ,-25 ,18 ,-21 ,-6 ,26 ,16 ,28 ,-18 ,-14 ,17 ,-25 ,29 ,1 ,17 ,4 ,19 ,-15 ,-18 ,-13 ,31 ,17 ,27 ,-2 ,-27 ,-31 ,8 ,16 ,26 ,17 ,29 ,-9 ,-13 ,-27 ,4 ,-7 ,31 ,11 ,18 ,18 ,27 ,-9 ,-3 ,-10 ,-26 ,-24 ,13 ,-4 ,5 ,-1 ,16 ,5 ,-29 ,-2 ,30 ,-29 ,23 ,8 ,-20 ,-29 ,-30 ,8 ,-20 ,-26 ,16 ,-6 ,25 ,-23 ,-7 ,21 ,-7 ,-10 ,-14 ,17 ,15 ,-16 ,-5 ,24 ,13 ,8 ,-30 ,11 ,4 ,-13 ,23 ,2 ,26 ,1 ,26 ,-3 ,-3 ,-10 ,-15 ,-22 ,18 ,-32 ,26};
int layer_1_bias[12] = {-13 ,-7 ,-20 ,20 ,-28 ,24 ,29 ,-16 ,21 ,-22 ,-2 ,-27};
int activate_layer_1( int x) { return x; } 
int layer_2_input[12];
int layer_2_output[5];
int layer_2_weights[60] = {35 ,-12 ,-11 ,7 ,-2 ,13 ,-16 ,0 ,-30 ,26 ,-35 ,-32 ,-6 ,-20 ,8 ,13 ,-3 ,18 ,20 ,26 ,-21 ,25 ,27 ,-22 ,-15 ,34 ,17 ,-28 ,-6 ,-16 ,26 ,32 ,-1 ,-31 ,1 ,-16 ,-4 ,29 ,24 ,14 ,30 ,20 ,8 ,32 ,21 ,12 ,-26 ,27 ,19 ,21 ,23 ,12 ,18 ,-18 ,3 ,16 ,22 ,16 ,-21 ,-5};
int layer_2_bias[5] = {-36 ,28 ,-24 ,-5 ,-6};
int activate_layer_2( int x) { return x; } 
int layer_3_input[5];
int layer_3_output[10];
int layer_3_weights[50] = {-39 ,-49 ,-32 ,-21 ,-21 ,25 ,26 ,-2 ,-8 ,-32 ,7 ,-29 ,43 ,-51 ,42 ,42 ,16 ,18 ,29 ,49 ,-1 ,-52 ,34 ,7 ,21 ,33 ,54 ,7 ,20 ,29 ,10 ,-19 ,42 ,41 ,45 ,-56 ,-11 ,9 ,-7 ,1 ,36 ,24 ,-28 ,-2 ,-36 ,-1 ,0 ,-14 ,-12 ,19};
int layer_3_bias[10] = {20 ,49 ,-11 ,-33 ,18 ,46 ,0 ,-16 ,-45 ,13};
int activate_layer_3( int x) { return x; } 
void deeplearn( hls::stream<intSdCh> &inStream, hls::stream<intSdCh> &outStream){

#pragma HLS INTERFACE axis port=outStream 

#pragma HLS INTERFACE axis port=inStream  

#pragma HLS INTERFACE ap_ctrl_none port=return 
#pragma HLS INTERFACE ap_ctrl_none port=return 
for (int idx = 0; idx < None; idx++){
			intSdCh valOut;
			intSdCh valIn = inStream.read();
			stream_input[idx] = valIn;
			input_matrix[idx] =valIn.data;
}
conv_layer_forward(input_matrix,layer_0_bias,layer_0_weights,layer_0_output,1,1);
dense_layer_forward(layer_1_input,layer_1_bias,layer_1_weights,output_matrix,16,12);
dense_layer_forward(layer_2_input,layer_2_bias,layer_2_weights,output_matrix,12,5);
dense_layer_forward(layer_3_input,layer_3_bias,layer_3_weights,layer_3_output,5,10);
for (int idx = 0; idx < 10; idx++){


			intSdCh valOut;
			intSdCh valIn;
			valIn = stream_input[idx];
			valIn.data = layer_1_output[idx];
			valOut.data = valIn.data ;

			valOut.keep = valIn.keep;
			valOut.strb = valIn.strb;
			valOut.user = valIn.user;
			valOut.last = valIn.last;
			

              if(idx==9){
				valOut.last = 1;
			}
			valOut.id = valIn.id;
			valOut.dest = valIn.dest;
			outStream.write(valOut);
}
}
