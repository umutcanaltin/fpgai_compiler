from pathlib import Path
from fpgai.util.fs import write_text

def _emit_hostsim_stubs(include_dir: Path) -> None:
    # Minimal: just drop stub headers into include/
    from fpgai.backends.hls.hostsim.ap_fixed_h import AP_FIXED_H
    from fpgai.backends.hls.hostsim.hls_stream_h import HLS_STREAM_H
    write_text(include_dir / "ap_fixed.h", AP_FIXED_H)
    write_text(include_dir / "hls_stream.h", HLS_STREAM_H)
