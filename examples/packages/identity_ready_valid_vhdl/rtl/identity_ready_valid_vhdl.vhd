library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity identity_ready_valid_vhdl is
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        input_valid : in std_logic;
        input_ready : out std_logic;
        input_data : in signed(15 downto 0);
        output_valid : out std_logic;
        output_ready : in std_logic;
        output_data : out signed(15 downto 0)
    );
end entity;

architecture rtl of identity_ready_valid_vhdl is
    signal full : std_logic := '0';
    signal data_reg : signed(15 downto 0) := (others => '0');
begin
    input_ready <= (not full) or output_ready;
    output_valid <= full;
    output_data <= data_reg;

    process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                full <= '0';
                data_reg <= (others => '0');
            elsif ((not full) or output_ready) = '1' then
                full <= input_valid;
                if input_valid = '1' then
                    data_reg <= input_data;
                end if;
            end if;
        end if;
    end process;
end architecture;
