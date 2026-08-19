library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity quantized_int8x4_split_vhdl is
  port (
    clk          : in  std_logic;
    rst_n        : in  std_logic;
    input_valid  : in  std_logic;
    input_ready  : out std_logic;
    input_data   : in  signed(31 downto 0);
    output_valid : out std_logic;
    output_ready : in  std_logic;
    main_data    : out signed(31 downto 0);
    skip_data    : out signed(31 downto 0)
  );
end entity;

architecture rtl of quantized_int8x4_split_vhdl is
  signal valid_q : std_logic := '0';
  signal data_q  : signed(31 downto 0) := (others => '0');
begin
  input_ready  <= (not valid_q) or output_ready;
  output_valid <= valid_q;
  main_data    <= data_q;
  skip_data    <= data_q;

  process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        valid_q <= '0';
        data_q  <= (others => '0');
      elsif ((not valid_q) or output_ready) = '1' then
        valid_q <= input_valid;
        if input_valid = '1' then
          data_q <= input_data;
        end if;
      end if;
    end if;
  end process;
end architecture;
