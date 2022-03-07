-- This is an oscilloscope VHDL model
-- Initially written by Marko Kosunen
-- Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 08.08.2017 11:41
LIBRARY ieee;
USE ieee.std_logic_1164.all;
USE ieee.numeric_std.all;
USE std.textio.all;


ENTITY oscilloscope IS
    PORT( A : IN  STD_LOGIC;
          Z : OUT STD_LOGIC
        );
END oscilloscope;

ARCHITECTURE rtl OF oscilloscope IS
BEGIN
    invert:PROCESS(A)
    BEGIN
        Z<=NOT A;
    END PROCESS;
END ARCHITECTURE;

