library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity scale_bias_vhdl is
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        input_valid : in std_logic;
        input_data : in signed(15 downto 0);
        output_valid : out std_logic;
        output_data : out signed(15 downto 0)
    );
end entity;

architecture rtl of scale_bias_vhdl is
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                output_valid <= '0';
                output_data <= (others => '0');
            else
                output_valid <= input_valid;
                output_data <= input_data;
            end if;
        end if;
    end process;
end architecture;
