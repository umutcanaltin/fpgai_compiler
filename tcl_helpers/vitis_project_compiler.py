import subprocess
import os
def create_vitis_tcl_file(tcl_filename,project_name,project_dir,main_files,test_file,file_location,top_function = "deeplearn",solution_name = "solution1"
                          ,part_name = "xc7z020clg400-1",clock_period = "10"):
    # Define the content of the TCL script
    tcl_content = 'set project_name "' + project_name +'"'
    tcl_content += 'set project_dir "' + project_dir +'"'
    for i in range(len(main_files)):
        tcl_content += 'set '+main_files[i]+' "' + file_location+'/'+ main_files[i] +'"'
    tcl_content += 'set '+test_file+' "' + file_location+'/'+test_file +'"'
    tcl_content += 'file mkdir $project_dir \n cd $project_dir \n open_project $project_name'
    for i in range(len(main_files)):
        tcl_content += 'add_files $'+main_files[i]
    tcl_content += 'add_files $'+test_file
    tcl_content += 'set_top '+top_function
    tcl_content += 'open_solution '+solution_name
    tcl_content += 'set_part '+part_name
    tcl_content += 'create_clock -period '+clock_period + " -name default"
    tcl_content += 'csim_design \n csynth_design \n cosim_design -rtl verilog \n export_design -rtl verilog -format ip_catalog \n exit'

    # Write the TCL script to a file
    with open(tcl_filename, 'w') as tcl_file:
        tcl_file.write(tcl_content)

    print(f"TCL file '{tcl_filename}' created successfully.")
    return os.path.abspath(tcl_filename)


def create_vitis_tcl_file_for_simulation(tcl_filename,project_name,project_dir,main_files,test_file,file_location,top_function = "deeplearn",solution_name = "solution1"
                          ,part_name = "xc7z020clg400-1",clock_period = "10"):
    # Define the content of the TCL script
    tcl_content = 'set project_name "' + project_name +'"\n'
    tcl_content += 'set project_dir "' + project_dir +'"\n'
    for i in range(len(main_files)):
        tcl_content += 'set '+main_files[i]+' "' + file_location+'/'+ main_files[i] +'"\n'
    tcl_content += 'set '+test_file+' "' + file_location+'/'+test_file +'"\n'
    tcl_content += 'file mkdir $project_dir \n cd $project_dir \n open_project $project_name \n'
    for i in range(len(main_files)):
        tcl_content += 'add_files '+file_location+"/"+main_files[i] + "\n"
    tcl_content += 'add_files '+file_location+"/"+test_file+ " -tb \n"
    tcl_content += 'set_top '+top_function+ "\n"
    tcl_content += 'open_solution '+solution_name+ "\n"
    tcl_content += 'set_part '+part_name+ "\n"
    tcl_content += 'create_clock -period '+clock_period + " -name default"+ "\n"
    tcl_content += 'csim_design \n exit'+ "\n"

    # Write the TCL script to a file
    with open(tcl_filename, 'w') as tcl_file:
        tcl_file.write(tcl_content)

    print(f"TCL file '{tcl_filename}' created successfully.")
    return os.path.abspath(tcl_filename)


def run_tcl_with_sourced_env(vitis_path, tcl_script_path):

    cmd = vitis_path

    hls_proc = subprocess.Popen("source settings64.sh ; vitis_hls  -f "+tcl_script_path, shell=True, executable="/bin/bash", cwd=cmd)
    out, err = hls_proc.communicate()




