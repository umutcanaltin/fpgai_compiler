def write_cpp_file(file_name,string):
    f = open(file_name + ".cpp", "a")
    f.write(string)
    f.close()
