set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        set project_name compiler_vivado_project
set project_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/
set ip_repo_dir /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files/compiler_hls_project/solution1/impl/ip
set top_module "deeplearn_0"
        set part_name "xc7z020clg400-1" 


        # Create a new Vivado project
        create_project $project_name $project_dir -part $part_name

        set_property ip_repo_paths [list $ip_repo_dir] [current_project]
        update_ip_catalog

        # Create block design
        create_bd_design "deeplearn_bd"
        # Create interface ports
        set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
        set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]
        # Create instance: processing_system7_0, and set properties
        set processing_system7_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0 ]
        set_property -dict [list         CONFIG.PCW_FPGA_FCLK0_ENABLE {1}         CONFIG.PCW_IRQ_F2P_INTR {1}         CONFIG.PCW_USE_FABRIC_INTERRUPT {1}         CONFIG.PCW_USE_S_AXI_HP0 {1}         ] $processing_system7_0

        # Create instance: deeplearn_0, and set properties
        set deeplearn_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:deeplearn:1.0 deeplearn_0 ]
        # Create instance: ps7_0_axi_periph, and set properties
        set ps7_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph ]
        set_property CONFIG.NUM_MI {2} $ps7_0_axi_periph

        # Create instance: rst_ps7_0_50M, and set properties
        set rst_ps7_0_50M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_50M ]

        # Create instance: axi_dma_0, and set properties
        set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
        set_property -dict [list         CONFIG.c_include_mm2s_dre {1}         CONFIG.c_include_s2mm_dre {1}         CONFIG.c_include_sg {0}         CONFIG.c_sg_length_width {23}         ] $axi_dma_0

        # Create instance: axi_mem_intercon, and set properties
        set axi_mem_intercon [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon ]
        set_property -dict [list         CONFIG.NUM_MI {1}         CONFIG.NUM_SI {2}         ] $axi_mem_intercon

        # Create instance: xlconcat_0, and set properties
        set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
        set_property CONFIG.NUM_PORTS {3} $xlconcat_0

        # Create interface connections
        connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins deeplearn_0/inStream] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_intercon/S00_AXI]
        connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_intercon/S01_AXI]
        connect_bd_intf_net -intf_net axi_mem_intercon_M00_AXI [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
        connect_bd_intf_net -intf_net deeplearn_0_outStream [get_bd_intf_pins deeplearn_0/outStream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
        connect_bd_intf_net -intf_net processing_system7_0_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins processing_system7_0/DDR]
        connect_bd_intf_net -intf_net processing_system7_0_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins processing_system7_0/FIXED_IO]
        connect_bd_intf_net -intf_net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M00_AXI [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins deeplearn_0/s_axi_control]
        connect_bd_intf_net -intf_net ps7_0_axi_periph_M01_AXI [get_bd_intf_pins ps7_0_axi_periph/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

        # Create port connections
        connect_bd_net -net axi_dma_0_mm2s_introut [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
        connect_bd_net -net axi_dma_0_s2mm_introut [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
        connect_bd_net -net deeplearn_0_interrupt [get_bd_pins deeplearn_0/interrupt] [get_bd_pins xlconcat_0/In2]
        connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins ps7_0_axi_periph/S00_ACLK] [get_bd_pins rst_ps7_0_50M/slowest_sync_clk] [get_bd_pins deeplearn_0/ap_clk] [get_bd_pins ps7_0_axi_periph/M00_ACLK] [get_bd_pins ps7_0_axi_periph/ACLK] [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins ps7_0_axi_periph/M01_ACLK] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_mem_intercon/S00_ACLK] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins axi_mem_intercon/M00_ACLK] [get_bd_pins axi_mem_intercon/ACLK] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] [get_bd_pins axi_mem_intercon/S01_ACLK]
        connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_50M/ext_reset_in]
        connect_bd_net -net rst_ps7_0_50M_peripheral_aresetn [get_bd_pins rst_ps7_0_50M/peripheral_aresetn] [get_bd_pins ps7_0_axi_periph/S00_ARESETN] [get_bd_pins deeplearn_0/ap_rst_n] [get_bd_pins ps7_0_axi_periph/M00_ARESETN] [get_bd_pins ps7_0_axi_periph/ARESETN] [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins ps7_0_axi_periph/M01_ARESETN] [get_bd_pins axi_mem_intercon/S00_ARESETN] [get_bd_pins axi_mem_intercon/M00_ARESETN] [get_bd_pins axi_mem_intercon/ARESETN] [get_bd_pins axi_mem_intercon/S01_ARESETN]
        connect_bd_net -net xlconcat_0_dout [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

        # Create address segments
        assign_bd_address -offset 0x41E00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
        assign_bd_address -offset 0x40000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs deeplearn_0/s_axi_control/Reg] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force
        assign_bd_address -offset 0x00000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -force

        validate_bd_design
        save_bd_design

        set_property source_mgmt_mode None [current_project] 

        make_wrapper -files [get_files /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.srcs/sources_1/bd/deeplearn_bd/deeplearn_bd.bd] -top
add_files -norecurse /home/umutcanaltin/Desktop/github_projects/fpgai_compilergenerated_files//compiler_vivado_project.gen/sources_1/bd/deeplearn_bd/hdl/deeplearn_bd_wrapper.v

        set_property top deeplearn_bd_wrapper [current_fileset]

        # Reset previous runs
        reset_run synth_1
        reset_run impl_1
        # Launch synthesis
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1

        # Launch implementation
        launch_runs impl_1 -jobs 4
        wait_on_run impl_1

        launch_runs impl_1 -to_step write_bitstream -jobs 10


        