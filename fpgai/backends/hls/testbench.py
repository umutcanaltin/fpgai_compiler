from __future__ import annotations
from pathlib import Path

def emit_tb_cpp(tb_dir: Path, *, top_name: str, in_words: int, out_words: int, weights_mode: str) -> None:
    tb_path = tb_dir / "tb.cpp"
    tb_path.write_text(
        f"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

typedef ap_axis<32,0,0,0> axis_t;

extern "C" void {top_name}(hls::stream<axis_t>& in, hls::stream<axis_t>& out);

int main(int argc, char** argv) {{
    // 1. Determine Input File
    const char* in_path = "input.bin";
    if (argc >= 2) in_path = argv[1];

    std::ifstream f(in_path, std::ios::binary);
    if (!f) {{
        fprintf(stderr, "[TB] Error: Could not open input file: %s\\n", in_path);
        return 1;
    }}
    
    // Read all floats
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    int n_floats = size / sizeof(float);
    
    std::vector<float> input_data(n_floats);
    f.read(reinterpret_cast<char*>(input_data.data()), size);
    f.close();

    printf("[TB] Loaded %d inputs from %s\\n", n_floats, in_path);

    // 2. Push to Stream
    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;

    for(int i=0; i<n_floats; i++) {{
        union {{ float f; unsigned int i; }} u;
        u.f = input_data[i];
        
        axis_t pkt;
        pkt.data = u.i;
        pkt.keep = -1;
        pkt.strb = -1;
        pkt.last = (i == n_floats - 1);
        in_stream.write(pkt);
    }}

    // 3. Run HLS Kernel
    printf("[TB] Running {top_name}...\\n");
    {top_name}(in_stream, out_stream);

    // 4. Read Output
    std::vector<float> output_data;
    
    while(!out_stream.empty()) {{
        axis_t pkt = out_stream.read();
        union {{ unsigned int i; float f; }} u;
        u.i = pkt.data.to_uint();
        output_data.push_back(u.f);
    }}
    
    printf("[TB] Received %zu outputs.\\n", output_data.size());

    // 5. Write Binary Output
    std::ofstream of("output.bin", std::ios::binary);
    of.write(reinterpret_cast<const char*>(output_data.data()), output_data.size() * sizeof(float));
    of.close();
    printf("[TB] Wrote output.bin\\n");

    return 0;
}}
""",
        encoding="utf-8",
    )