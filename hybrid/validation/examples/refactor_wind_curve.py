import hybrid.validation.wind.iessGE15.m5_wind_resource as m5

start = 'August 2012'
end = 'October 2022'

# m5.import_m5_wind(start_month=start, end_month=end)
bin_res = 2
# m5.downsample_m5_wind(bin_resolution=bin_res, start_month=start, end_month=end)
# m5.plot_m5_wind_bins(bin_resolution=bin_res, start_month=start, end_month=end)
power_curve = "NREL_Reference_1.5MW_Turbine_Site_Level.csv"
rotor_diameter = 77
rho = 1.000
m5.refactor_power_curve_m5_wind(power_curve, rotor_diameter, air_density=rho, bin_resolution=bin_res, start_month=start, end_month=end)