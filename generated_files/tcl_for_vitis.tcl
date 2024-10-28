set project_name compiler_hls_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/generated_files/

    file mkdir $project_dir
    cd $project_dir
    open_project $project_name
    add_files /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/generated_files/main.cpp
add_files /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/generated_files/testbench.cpp
add_files /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/generated_files/deeplearn.h
set_top deeplearn
    open_solution "solution1"
    set_part xc7z020clg400-1

    create_clock -period 10 -name default

    # Run C Simulation to validate the behavior of the design with the testbench
    csim_design

    # Run C Synthesis
    csynth_design

    # Run C/RTL co-simulation to verify the generated RTL matches the C design
    cosim_design -rtl verilog

    # Generate RTL code in Verilog format, ready for IP catalog
    export_design -rtl verilog -format ip_catalog

    # Exit the HLS tool
    exit

    