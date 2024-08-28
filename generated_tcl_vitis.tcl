set project_name "generated_project"
set project_dir "/home/umutcanaltin/Desktop/generated_dir"
set deeplearn.h "/home/umutcanaltin/Desktop/tcl_scripts/deeplearn.h"
set main.cpp "/home/umutcanaltin/Desktop/tcl_scripts/main.cpp"
set testbench.cpp "/home/umutcanaltin/Desktop/tcl_scripts/testbench.cpp"
file mkdir $project_dir 
 cd $project_dir 
 open_project $project_name 
add_files /home/umutcanaltin/Desktop/tcl_scripts/deeplearn.h
add_files /home/umutcanaltin/Desktop/tcl_scripts/main.cpp
add_files /home/umutcanaltin/Desktop/tcl_scripts/testbench.cpp -tb 
set_top deeplearn
open_solution solution1
set_part xc7z020clg400-1
create_clock -period 10 -name default
csim_design 
 exit
