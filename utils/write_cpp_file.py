def write_cpp_file(file_name,string):
    f = open(file_name + ".cpp", "a")
    f.write(string)
    f.close()

def write_header_file(file_name,string):
    f = open(file_name + ".h", "a")
    f.write(string)
    f.close()

def write_tcl_file(file_name,string):
    f = open(file_name + ".tcl", "a")
    f.write(string)
    f.close()