#include <math.h>
#include <stdio.h>

void conv2d(float * input, float  * filters, int filter_number, int filter_size_1, int filter_size_2,int input_size_1, int input_size_2, float * output){
    for(int filter_index = 0 ; filter_index < filter_number ;  filter_index++ ){
        // select individual filter
        for(int conv_x_index=0 ; conv_x_index < input_size_1 - filter_size_1 + 1 ; conv_x_index ++ ){ 
            for(int conv_y_index=0 ; conv_y_index < input_size_2 - filter_size_2 + 1 ; conv_y_index ++ ){
            // stride kernel in input 
            float conv_sum = 0;

                for(int kernel_x_index=0 ; kernel_x_index < filter_size_1 ; kernel_x_index ++ ){  
                    for(int kernel_y_index =0; kernel_y_index < filter_size_2 ; kernel_y_index++){ 
                        conv_sum += *(filters + kernel_y_index + filter_size_2*kernel_x_index)* *(input + conv_y_index + kernel_y_index + input_size_2*kernel_x_index);
                        
                    } 
                }
            *(output + conv_y_index  + conv_x_index*input_size_2) = conv_sum; 
            }
        }
    } 
}

void pixel_shufle(float * input, int input_size_1, int input_size_2, int input_size_3,  float * output){
    // input_size_2 should be equal to input_size_3
    
    for(int input_x = 0; input_x< input_size_2; input_x++){
        for(int input_y = 0; input_y< input_size_3; input_y++){
            for(int input_id = 0; input_id< input_size_1; input_id++){
                if(input_id < sqrt(input_size_1)){
                    *(output + input_id + input_y*sqrt(input_size_1) + input_x*input_size_1*input_size_3);
                }
                else{
                    *(output + input_id + input_y*sqrt(input_size_1)  + input_x*input_size_1*input_size_3 + input_size_3*sqrt(input_size_1) );
                }
                

            }
        }
    }
}