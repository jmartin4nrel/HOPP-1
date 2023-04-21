# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:11:00 2023

@author: emendiol
"""
from influxdb import InfluxDBClient
from datetime import datetime,timedelta
from dateutil import tz,parser
import time
import pytz
import csv

#function to convert datetime to Unix number in milliseconds
def getUnix(t):
    tzone = pytz.timezone('America/Denver')
    t = tzone.localize(t)
    t.astimezone(tz.gettz('UTC'))
    t_unix = int(round((t - datetime(1970,1,1,tzinfo=tz.gettz('UTC'))).total_seconds()*1000))
    return t_unix


#start InfluxDB client with udp database
Influx_client = InfluxDBClient(host='10.20.5.158',port=8086,username='hopp',password='hopp4321!',database='udp')

#end time of 12:05:00PM for June 25,2022
tend = datetime(2022,7,25,12,5,0)
#start time will be 5 minutes before end time
tstart = tend - timedelta(minutes=5)


#convert to unix time for InfluxDB query
tendUnix = str(getUnix(tend)) + 'ms'
tstartUnix = str(getUnix(tstart)) + 'ms'


#######################################################################################################################################################
#################################################################### Measured Data #################################################################### 
#######################################################################################################################################################
####################################### Active Power #######################################
Influx_client.switch_database('udp')
#form query string for InfluxDB
query = '''SELECT first("PQ_F17_Pactive") FROM "processed_data" WHERE ("node" = 'PQ_F17') AND time >= ''' + tstartUnix + ' AND time <= ' + tendUnix + ' GROUP BY time(1s) fill(previous)'
#you can also use first ,mean ,max ,min instead of first for selecting the measurement signal name
#For the fill filter at the end of the query, you can also input null, none ,0 , or linear in instead of previous

#query = 'SELECT'' first("PQ_F18_Pactive") FROM "processed_data" WHERE ("node" = ' + "'PQ_F18'" + ') AND time >= ' + tstartUnix + ' AND time <= ' + tendUnix + ' GROUP BY time(1s) fill(previous)'
#query = 'SELECT'' first("PQ_GE15_Pactive") FROM "processed_data" WHERE ("node" = ' + "'PQ_GE15'" + ') AND time >= ' + tstartUnix + ' AND time <= ' + tendUnix + ' GROUP BY time(1s) fill(previous)'
#query InfluxDB
res = Influx_client.query(query)
#extract data from query result
#data is returned with timestamp for each point. Timestamp is in UTC 
data = ((((res.raw)['series'])[0])['values'])
#power data is returned in units of watts

####################################### Measured Irradiance #######################################
#Measured irradiance data fomes from reference panel in First Solar 430kW PV array 
query = '''SELECT first("irradiance") FROM "weather" WHERE ("site" = 'FirstSolar') AND time >= ''' + tstartUnix + ' AND time <= ' + tendUnix + ' GROUP BY time(1s) fill(previous)'



#######################################################################################################################################################
#################################################################### Forecast Data #################################################################### 
#######################################################################################################################################################
#Switch database to Weather_Forecast
Influx_client.switch_database('Weather_Forecast')

####################################### Query Hour Ahead #######################################
#HourAhead2 forecast has the following signals available: DIFI, DNI, GHI, PressureHPA, RelativeHumidity, TemperatureC, WindDirection, WindSpeed
#hour filter ranges from hr00 to hr23 and minute filter ranges from min00 to min59
query = '''SELECT first("WindSpeed") FROM "HourAhead2" WHERE ("hour" = 'hr01' AND "minute" = 'min01') AND time >= ''' + tstartUnix + ' AND time <= ' + tendUnix + ' GROUP BY time(1s) fill(previous)'

####################################### Query Day Ahead #######################################
#DayAhead2 forecast has the following signals available: DIFI, DNI, GHI, PressureHPA, RelativeHumidity, TemperatureC, WindDirection, WindSpeed
#day filter ranges from day00 to day06 
query = '''SELECT first("WindSpeed") FROM "DayAhead2" WHERE ("day" = 'day01') AND time >= ''' + tstartUnix + ' AND time <= ' + tendUnix + ' GROUP BY time(1s) fill(previous)'

####################################### Query Week Ahead #######################################
#WeekForecast2 forecast has the following signals available: cloud_amount, hourly_precipitation, humidity, precipitation_probability, temperature_dew_point, temperature_hourly, temperature_wind_chill, wind_direction, wind_speed_gust, wind_speed_sustained
#day filter ranges from hr000 to hr167 
query = '''SELECT first("wind_speed_sustained") FROM "WeekForecast2" WHERE ("hour" = 'hr001') AND time >= ''' + tstartUnix + ' AND time <= ' + tendUnix + ' GROUP BY time(1s) fill(previous)'



