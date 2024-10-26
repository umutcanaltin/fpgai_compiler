def vitis_tcl_generator(src_file_dir,project_name , project_dir):
    vitis_str= "set project_name " + project_name + "\n"
    vitis_str+="set project_dir " + project_dir + "\n"
    vitis_str += """
    file mkdir $project_dir
    cd $project_dir
    open_project $project_name
    """
    vitis_str += "add_files " + src_file_dir +"/main.cpp\n"
    vitis_str += "add_files " + src_file_dir +"/testbench.cpp\n"
    vitis_str += "add_files " + src_file_dir +"/deeplearn.h\n"

    vitis_str += """set_top deeplearn
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

    """
    return vitis_str