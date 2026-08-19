library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity quantized_relu_int8x4_vhdl is
  port (
    clk          : in  std_logic;
    rst_n        : in  std_logic;
    input_valid  : in  std_logic;
    input_ready  : out std_logic;
    input_data   : in  signed(31 downto 0);
    output_valid : out std_logic;
    output_ready : in  std_logic;
    output_data  : out signed(31 downto 0)
  );
end entity;

architecture rtl of quantized_relu_int8x4_vhdl is
  signal valid_q : std_logic := '0';
  signal data_q  : signed(31 downto 0) := (others => '0');

  function relu_byte(x : signed(7 downto 0)) return signed is
  begin
    if x < 0 then
      return to_signed(0, 8);
    end if;
    return x;
  end function;
begin
  input_ready  <= (not valid_q) or output_ready;
  output_valid <= valid_q;
  output_data  <= data_q;

  process(clk)
    variable next_data : signed(31 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        valid_q <= '0';
        data_q  <= (others => '0');
      elsif ((not valid_q) or output_ready) = '1' then
        valid_q <= input_valid;
        if input_valid = '1' then
          next_data := (others => '0');
          for lane in 0 to 3 loop
            next_data((lane + 1) * 8 - 1 downto lane * 8) :=
              relu_byte(input_data((lane + 1) * 8 - 1 downto lane * 8));
          end loop;
          data_q <= next_data;
        end if;
      end if;
    end if;
  end process;
end architecture;
