# -*- coding: utf-8 -*-
"""
######################## Plant Controller Modbus Map ########################
################ Signals from HOPP to plant controller ################
#    holding register 0 -> year
#    holding register 1 -> month
#    holding register 2 -> day
#    holding register 3 -> hour
#    holding register 4 -> minute
#    holding register 5 -> second
#    holding register 6  -> BESS Pset (units of kW)
################ Signals to HOPP from plant controller ################
#    holding register 7 -> Bess Power (units of kW)
#    holding register 8 -> Bess SOC (divide by 10 to get SOC in percent)
#    holding register 9 -> Wind Power (units of kW)
#    holding register 10 -> PV Power (units of kW)
########################################################################
"""

from pymodbus.client.sync import ModbusTcpClient

mbClient = ModbusTcpClient(host='192.174.56.29',port=502)
connected = mbClient.connect()
res = mbClient.read_holding_registers(address=0, count=11)
#use the getRegisters function to decode a register value
year = res.getRegister(0)

#use the registers function to decode all the registers. Array with register values is returned
regs = res.registers

#use the write_register function to write a single value to one register. address specifies the starting modbus address
#in this example we set the year to zero on the modbus server
#Setting the year to zero will enable the plant controller to operate in real-time, using live PV,Wind power measurements
#Setting the year to non-zero value will enable the plant controller to oeperate from data based on the date provided on registers 0-5
mbClient.write_register(address=0, value=0)

#use the write_register function to write to multiple registers
#in this example we write the year,month,day,hour,minute,second, and power setpoint to the holding registers on the modbus server
# mbClient.write_registers(address=0,values=[2022, 8, 9, 15, 0, 0, 125])
