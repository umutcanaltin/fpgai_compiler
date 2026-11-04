# Read-only installed board definition discovery; no Vivado project or bitstream.
puts "FPGAI_PYNQ_Z2_INSTALLED_BOARD_PARTS_BEGIN"
set parts [get_board_parts -quiet *pynq-z2*]
foreach part [lsort $parts] {
  puts "BOARD_PART: $part"
}
if {[llength $parts] == 0} {
  puts "NO_PYNQ_Z2_BOARD_PART_INSTALLED"
}
puts "FPGAI_PYNQ_Z2_INSTALLED_BOARD_PARTS_END"
exit
