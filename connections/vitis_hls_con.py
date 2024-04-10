import os
import subprocess


class HLS_Connection():
    def __init__(self, config = None, data_test_file=None):
        self.tcl_file_name = "tcl_script.tcl"
        self.tcl_file = self.open_tcl_script_file(self.tcl_file_name)
        self.default_config = {
        "project_location": "",
        "vitis_hls_dir_name" : "/tools/Xilinx/Vitis_HLS/2023.2",
        "project_name" : "default_project",
        "top_level_function_name" : "main",
        "src_dir_name" : "src",
        "tb_dir_name" : "tb",
        "cflags": "",
        "src_files" : "main.cpp",
        "compiler": "",
        "tb_files" : "",
        "part_name" : "xc7z020clg400-1",
        "clock_period" : "",
        "language" : "vhdl"
        }
        self.config = self.default_config
        if(config!= None):
            self.config = config   
        self.set_config_to_tcl_file()

    def open_tcl_script_file(self, file_name):
        f = open(file_name, "w")
        return f
    
    def write_tcl_command_to_file(self,command):
        self.tcl_file.write(command)
    
    def close_tcl_file(self):
        self.tcl_file.close()
    
    def source_vitis_sh_file(self):
        cmd = "/tools/Xilinx/Vitis_HLS/2023.2"

    
        #foo= subprocess.Popen("ls", cwd=cmd)
        hls_proc = subprocess.Popen("source settings64.sh ; vitis_hls  -f /home/umutcanaltin/Desktop/github_projects/hls_connection/example.tcl", shell=True, executable="/bin/bash", cwd=cmd)
        out, err = hls_proc.communicate()
       

    
    def execute_command(self,command_to_execute):
        run = subprocess.run(command_to_execute, capture_output=True)
        return run


    def set_config_to_tcl_file(self):
        for config_item in self.config :
            if(config_item == "project_name"):
                self.write_tcl_command_to_file("open_project " + self.config[config_item] + "\n")
            if(config_item == "top_level_function_name"):
                self.write_tcl_command_to_file("set_top " + self.config[config_item]+ "\n")
            if(config_item == "src_files"):
                self.write_tcl_command_to_file("add_files " + self.config[config_item]+ "\n")
        self.write_tcl_command_to_file('open_solution "solution1" -flow_target vitis'+ "\n")
        for config_item in self.config :
            if(config_item == "part_name"):
                self.write_tcl_command_to_file("set_part {" + self.config[config_item] + "}" + "\n")
            if(config_item == "clock_period"):
                self.write_tcl_command_to_file("create_clock -period " + self.config[config_item] + "-name default"+ "\n")
        self.write_tcl_command_to_file('source "./dct/solution1/directives.tcl"'+ "\n")
        
        
        self.close_tcl_file()



from xml.dom.minidom import parse
class Vitis_HLS_Report_Parser():
    def __init__(self,report_files_path = "/home/umutcanaltin/Desktop/hls_projects/dma_array/solution1/syn/report/",csynth_file_name= "csynth.xml", csynth_design_size_file_name = "csynth_design_size.xml", main_func_name = "doGain"):
        self.csynth_file = parse(report_files_path+csynth_file_name)
        self.csynth_design_size_file = parse(report_files_path+csynth_design_size_file_name)
        self.main_func_file = parse(report_files_path+main_func_name + "_csynth.xml")
    
    def get_UserAssignments(self):
        user_assignments_dictionary = {}
        user_assignments = self.csynth_file.getElementsByTagName("UserAssignments")
        for assignment in user_assignments:
            user_assignments_dictionary["unit"] = assignment.getElementsByTagName("unit")[0].firstChild.data
            user_assignments_dictionary["ProductFamily"]  = assignment.getElementsByTagName("ProductFamily")[0].firstChild.data
            user_assignments_dictionary["Part"]  = assignment.getElementsByTagName("Part")[0].firstChild.data
            user_assignments_dictionary["TopModelName"]  = assignment.getElementsByTagName("TopModelName")[0].firstChild.data
            user_assignments_dictionary["TargetClockPeriod"] = assignment.getElementsByTagName("TargetClockPeriod")[0].firstChild.data
            user_assignments_dictionary["ClockUncertainty"]  = assignment.getElementsByTagName("ClockUncertainty")[0].firstChild.data
        return user_assignments_dictionary
    
    def get_PerformanceEstimates(self):
        PerformanceEstimates_dictionary = {}
        PerformanceEstimates = self.csynth_file.getElementsByTagName("PerformanceEstimates")
     
        for estimates in PerformanceEstimates:
       
            PerformanceEstimates_dictionary["PipelineType"] = estimates.getElementsByTagName("PipelineType")[0].firstChild.data
            
            SummaryOfTimingAnalysis = estimates.getElementsByTagName("SummaryOfTimingAnalysis")
            SummaryOfTimingAnalysis_dictionary = {}
            for _SummaryOfTimingAnalysis in SummaryOfTimingAnalysis:
                #PerformanceEstimates_dictionary["SummaryOfTimingAnalysis_unit"] = _SummaryOfTimingAnalysis.getElementsByTagName("unit")
                SummaryOfTimingAnalysis_dictionary["EstimatedClockPeriod"] = _SummaryOfTimingAnalysis.getElementsByTagName("EstimatedClockPeriod")[0].firstChild.data
            PerformanceEstimates_dictionary["SummaryOfTimingAnalysis"] =SummaryOfTimingAnalysis_dictionary

            SummaryOfOverallLatency = estimates.getElementsByTagName("SummaryOfOverallLatency")
            SummaryOfOverallLatency_dictionary = {}
            for _SummaryOfOverallLatency in SummaryOfOverallLatency:
                #PerformanceEstimates_dictionary["SummaryOfOverallLatency_unit"] = _SummaryOfOverallLatency.getElementsByTagName("unit")[0].firstChild.data
                SummaryOfOverallLatency_dictionary["Best-caseLatency"] = _SummaryOfOverallLatency.getElementsByTagName("Best-caseLatency")[0].firstChild.data
                SummaryOfOverallLatency_dictionary["Average-caseLatency"] = _SummaryOfOverallLatency.getElementsByTagName("Average-caseLatency")[0].firstChild.data
                SummaryOfOverallLatency_dictionary["Worst-caseLatency"] = _SummaryOfOverallLatency.getElementsByTagName("Worst-caseLatency")[0].firstChild.data
                SummaryOfOverallLatency_dictionary["Best-caseRealTimeLatency"] = _SummaryOfOverallLatency.getElementsByTagName("Best-caseRealTimeLatency")[0].firstChild.data
                SummaryOfOverallLatency_dictionary["Average-caseRealTimeLatency"] = _SummaryOfOverallLatency.getElementsByTagName("Average-caseRealTimeLatency")[0].firstChild.data
                SummaryOfOverallLatency_dictionary["Worst-caseRealTimeLatency"] = _SummaryOfOverallLatency.getElementsByTagName("Worst-caseRealTimeLatency")[0].firstChild.data
                #PerformanceEstimates_dictionary["SummaryOfOverallLatency_Interval-min"] = _SummaryOfOverallLatency.getElementsByTagName("Interval-min")[0].firstChild.data
                #PerformanceEstimates_dictionary["SummaryOfOverallLatency_Interval-max"] = _SummaryOfOverallLatency.getElementsByTagName("Interval-max")[0].firstChild.data
            PerformanceEstimates_dictionary["SummaryOfOverallLatency"] = SummaryOfOverallLatency_dictionary

            SummaryOfLoopLatency = estimates.getElementsByTagName("SummaryOfLoopLatency")
            SummaryOfLoopLatency_dictionary = {}
            for _SummaryOfLoopLatency in SummaryOfLoopLatency:
                for _SummaryOfLoopLatency_childNodes in _SummaryOfLoopLatency.childNodes[1:-1] :
                    SummaryOfLoopLatency_loop_dictionary = {}
                    SummaryOfLoopLatency_loop_dictionary["Slack"] = _SummaryOfLoopLatency_childNodes.getElementsByTagName("Slack")[0].firstChild.data
                    SummaryOfLoopLatency_loop_dictionary["TripCount"] = _SummaryOfLoopLatency_childNodes.getElementsByTagName("TripCount")[0].firstChild.data
                    SummaryOfLoopLatency_loop_dictionary["Latency"] = _SummaryOfLoopLatency_childNodes.getElementsByTagName("Latency")[0].firstChild.data
                    SummaryOfLoopLatency_loop_dictionary["AbsoluteTimeLatency"] = _SummaryOfLoopLatency_childNodes.getElementsByTagName("AbsoluteTimeLatency")[0].firstChild.data
                    SummaryOfLoopLatency_loop_dictionary["PipelineII"] = _SummaryOfLoopLatency_childNodes.getElementsByTagName("PipelineII")[0].firstChild.data
                    SummaryOfLoopLatency_loop_dictionary["PipelineDepth"] = _SummaryOfLoopLatency_childNodes.getElementsByTagName("PipelineDepth")[0].firstChild.data
                    SummaryOfLoopLatency_dictionary[_SummaryOfLoopLatency_childNodes.nodeName] =SummaryOfLoopLatency_loop_dictionary

            PerformanceEstimates_dictionary["SummaryOfLoopLatency"] = SummaryOfLoopLatency_dictionary
            return PerformanceEstimates_dictionary
        
    def get_AreaEstimates(self):
        AreaEstimates_dictionary = {}
        AreaEstimates = self.csynth_file.getElementsByTagName("AreaEstimates")

        for estimates in AreaEstimates:
            Resources = estimates.getElementsByTagName("Resources")
            Resources_dictionary = {}
            for _Resources in Resources:
                Resources_dictionary["BRAM_18K"] = _Resources.getElementsByTagName("BRAM_18K")[0].firstChild.data
                Resources_dictionary["DSP"] = _Resources.getElementsByTagName("DSP")[0].firstChild.data
                Resources_dictionary["FF"] = _Resources.getElementsByTagName("FF")[0].firstChild.data
                Resources_dictionary["LUT"] = _Resources.getElementsByTagName("LUT")[0].firstChild.data
                Resources_dictionary["URAM"] = _Resources.getElementsByTagName("URAM")[0].firstChild.data
                AreaEstimates_dictionary["Resources"] = Resources_dictionary

            AvailableResources = estimates.getElementsByTagName("AvailableResources")
            AvailableResources_dictionary = {}
            for _AvailableResources in AvailableResources:
                AvailableResources_dictionary["BRAM_18K"] = _AvailableResources.getElementsByTagName("BRAM_18K")[0].firstChild.data
                AvailableResources_dictionary["DSP"] = _AvailableResources.getElementsByTagName("DSP")[0].firstChild.data
                AvailableResources_dictionary["FF"] = _AvailableResources.getElementsByTagName("FF")[0].firstChild.data
                AvailableResources_dictionary["LUT"] = _AvailableResources.getElementsByTagName("LUT")[0].firstChild.data
                AvailableResources_dictionary["URAM"] = _AvailableResources.getElementsByTagName("URAM")[0].firstChild.data
                AreaEstimates_dictionary["AvailableResources"] = AvailableResources_dictionary
            return AreaEstimates_dictionary
                    



            


