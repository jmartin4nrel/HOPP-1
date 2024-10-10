from greenheart.simulation.technologies.hydrogen.electrolysis.ALK_electrolyzer_clusters import ALK_Clusters 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
def estimate_alkaline_footprint_singlitico(electrolyzer_size_MW):
    #https://doi.org/10.1016/j.rset.2021.100005
    footprint_m2_pr_MW = 95
    footprint_m2 = footprint_m2_pr_MW*electrolyzer_size_MW
    return footprint_m2

def estimate_alkaline_system_footprint(n_stacks):
    #Thomas I. Valdez, ... Sebastian Freund, in Machinery and Energy Systems for the Hydrogen Economy, 2022
    #https://www.sciencedirect.com/topics/engineering/alkaline-electrolysis
    # 5.4.7.2: Commercial alkaline electrolysis systems
    # 0.5 MW electrolyzer: 2.2m x 2.3 m by 3m tall
    #Teledyne Titan EL water electrolysis system in 80 Nm3/h configuration.
    #assuming double the volume of 0.5 MW electrolyzer and square shaped
    ref_stack_width = 5.06 #m
    ref_stack_length = 5.06 #m
    ref_stack_height = 3 #m
    #https://www.sinohyenergy.com/5mw-10mw-alkaline-water-electrolysis/
    #1800 L H2O/hour, 2000 Nm^3-H2/hr, 1000 Nm^3 O2/hr
    system_width = n_stacks*ref_stack_width
    system_length = n_stacks*ref_stack_length
    system_area = system_width*system_length
    system_volume = system_area*ref_stack_height
    return {"System Width [m]":system_width,"System Length [m]":system_length,"System Area [m^2]":system_area,"System Volume [m^3]":system_volume}

def gibbs_temp_pressure(T_c):
    #https://webbook.nist.gov/cgi/inchi/InChI%3D1S/H2O/h1H2
    #^ shomate equatiom
    T_k=T_c+ 273.15  # convert Celsius to Kelvin
    t = T_k/1000
    enthalpy_stc = -285.83 #kJ/mol
    entropy = 69.95 #J/mol-K
    a = -203.6060
    b = 1523.290
    c = -3196.413
    d = 2474.455
    e = 3.855326
    f = -256.5478
    g = -488.7163
    h = -285.8304

    delta_h = a*t + b*(t**2)/2 + c*(t**3)/3 + d*(t**4)/4 - (e/t) + f - h
    s = a*np.log(t) + b*t + c*(t**2)/2 + d*(t**3)/3 - e/(2*(t**2)) + g

    pass
def get_efficiency_curve(alk: ALK_Clusters,file_desc = "test"):
    filepath = os.path.join(os.path.dirname(__file__),"Alkaline_Efficiency-Curve-{}.csv".format(file_desc))
    dA = 10
    current_range = np.arange(dA,alk.nominal_current+dA,dA) 
    current_density = np.zeros(len(current_range))
    V_cell = np.zeros(len(current_range))
    H2_cell = np.zeros(len(current_range)) #kg/hr

    for ii,I_stack in enumerate(current_range):
        current_density[ii] = alk.calc_current_density(alk.T_stack, I_stack)
        V_cell[ii] = alk.cell_design(alk.T_stack,I_stack)
        H2_cell[ii] = alk.cell_H2_production_rate(alk.T_stack,I_stack)
    H2_stack = H2_cell*alk.n_cells
    Stack_Power_kW = current_range*V_cell*alk.n_cells/1e3
    power_usage_kWh = current_range*V_cell/1e3
    efficiency_kWh_pr_kg = power_usage_kWh/H2_cell
    keys = ["I [A]","V_cell [V]","Stack Power [kW]","Stack H2 Production [kg/hr]","Efficiency [kWh/kg]","Load %"]
    vals = [current_range,V_cell,Stack_Power_kW,H2_stack,efficiency_kWh_pr_kg,100*Stack_Power_kW/alk.stack_rating_kW]
    df = pd.DataFrame(dict(zip(keys,vals)))
    df.to_csv(filepath)
    print("Saved efficiency curve to: {}".format(filepath))
    return df
def plot_IV_curve(alk:ALK_Clusters,file_desc = "test"):
    dA = 10
    # current_range = np.arange(alk.min_current,alk.nominal_current+dA,dA) 
    current_range = np.arange(dA,alk.nominal_current+dA,dA) 
    current_density = np.zeros(len(current_range))
    V_cell = np.zeros(len(current_range))

    U_rev = np.zeros(len(current_range))
    V_ohm = np.zeros(len(current_range))
    
    V_act_a = np.zeros(len(current_range))
    V_act_c = np.zeros(len(current_range))

    V_ohm_electrode = np.zeros(len(current_range))
    V_ohm_electrolyte = np.zeros(len(current_range))
    V_ohm_membrane = np.zeros(len(current_range))

    theta = np.zeros(len(current_range))
    epsilon = np.zeros(len(current_range))

    for ii,I_stack in enumerate(current_range):
        theta[ii],epsilon[ii] = alk.cell_bubble_rate_coverage(alk.T_stack, I_stack)
        current_density[ii] = alk.calc_current_density(alk.T_stack, I_stack)
        V_cell[ii] = alk.cell_design(alk.T_stack,I_stack)

        U_rev[ii] = alk.cell_reversible_overpotential(alk.T_stack, alk.pressure_operating)
        V_ohm[ii] = alk.cell_ohmic_overpotential(alk.T_stack, I_stack)
        V_act_a[ii], V_act_c[ii] = alk.cell_activation_overpotential(alk.T_stack, I_stack)
        

        R_a, R_c = alk.cell_electrode_resistance(alk.T_stack)
        R_ele_bf, R_ele_b = alk.cell_electrolyte_resistance(alk.T_stack, I_stack)  # [Ohms]
        R_membrane = alk.cell_membrane_resistance(alk.T_stack)  # [Ohms] 

        V_ohm_electrode[ii] = I_stack*(R_a + R_c)
        V_ohm_electrolyte[ii] = I_stack*(R_ele_bf + R_ele_b)
        V_ohm_membrane[ii] = I_stack*(R_membrane)

    keys = ["I [A]","J [A/cm^2]","V_cell","U_rev","V_act,a","V_act,c","V_ohm,a-c","V_ohm,KOH","V_ohm,mem","Fractional Bubble Coverage","Bulk Bubbling Coeff"]
    vals = [current_range,current_density,V_cell,U_rev,V_act_a,V_act_c,V_ohm_electrode,V_ohm_electrolyte,V_ohm_membrane,theta,epsilon]
    df = pd.DataFrame(dict(zip(keys,vals)))
    df.to_csv(os.path.join(os.path.dirname(__file__),"Alkaline_IV-Curve_Data-{}.csv".format(file_desc)))

    fig, ax = plt.subplots(figsize=[8,8])
    
    horiz_al = "left"
    vert_al = "center"
    s_font = "medium"#"small"
    w_font = "semibold"
    text_props = {"ha":horiz_al,"va":vert_al,"fontsize":s_font,"fontweight":w_font}
    alpha_fill = 0.5
    v_lw = 1.5
    v_ls = "solid"
    xtext = np.round(np.max(current_density),1)
    #V_act os gold. V_ohm is green, U_rev is grey

    v_color_line = "grey"
    v_color_fill = "lightgrey"
    y_lb = np.zeros(len(current_density))
    y_ub = U_rev
    y_mid = np.max(y_lb + (y_ub-y_lb)/2)
    # ax.plot(current_density,U_rev,color=v_color_line,lw=v_lw,ls=v_ls,label="$U_{rev}$")
    # ax.fill_between(current_density,np.zeros(len(current_density)),U_rev,color=v_color_fill,alpha=alpha_fill)
    ax.plot(current_density,y_ub,color=v_color_line,lw=v_lw,ls=v_ls,label="$U_{rev}$")
    ax.fill_between(current_density,y_lb,y_ub,color=v_color_fill,alpha=alpha_fill)
    ax.text(x=xtext,y=y_mid,s="$U_{rev}$",color=v_color_line,**text_props)
    
    #V-act
    y_lb+=U_rev
    y_ub+=V_act_a
    y_mid = np.max(y_lb + (y_ub-y_lb)/2)
    v_lw = 0.75
    v_color_line = "darkorange"
    v_color_fill = "gold"
    v_ls = "--"
    # ax.plot(current_density,U_rev+V_act_a,color=v_color_line,lw=v_lw,ls=v_ls,label="$V_{act,a}$")
    # ax.fill_between(current_density,U_rev,U_rev+V_act_a,color=v_color_fill,alpha=alpha_fill)
    ax.plot(current_density,y_ub,color=v_color_line,lw=v_lw,ls=v_ls,label="$V_{act,a}$")
    ax.fill_between(current_density,y_lb,y_ub,color=v_color_fill,alpha=alpha_fill)
    ax.text(x=xtext,y=y_mid,s="$V_{act,a}$",color=v_color_line,**text_props)

    y_lb+=V_act_a
    y_ub+=V_act_c
    y_mid = np.max(y_lb + (y_ub-y_lb)/2)
    v_lw = 1.5
    v_color_line = "orange"
    v_color_fill = "goldenrod"
    v_ls = "solid"
    # ax.plot(current_density,U_rev+V_act_a+V_act_c,color=v_color_line,lw=v_lw,ls=v_ls,label="$V_{act,c}$")
    # ax.fill_between(current_density,U_rev+V_act_a,U_rev+V_act_a+V_act_c,color=v_color_fill,alpha=alpha_fill)
    ax.plot(current_density,y_ub,color=v_color_line,lw=v_lw,ls=v_ls,label="$V_{act,c}$")
    ax.fill_between(current_density,y_lb,y_ub,color=v_color_fill,alpha=alpha_fill)
    ax.text(x=xtext,y=y_mid,s="$V_{act,c}$",color=v_color_line,**text_props)

    #V-Ohm
    y_lb+=V_act_c
    y_ub+=V_ohm_electrode
    y_mid = np.max(y_lb + (y_ub-y_lb)/2)
    v_lw = 0.75
    v_color_line = "green"
    v_color_fill = "palegreen"
    v_ls = "--"
    # ax.plot(current_density,U_rev+V_act_a+V_act_c+V_ohm_electrode,color=v_color_line,lw=v_lw,ls=v_ls,label="$V_{\Omega,a-c}$")
    # ax.fill_between(current_density,U_rev+V_act_a+V_act_c,U_rev+V_act_a+V_act_c+V_ohm_electrode,color=v_color_fill,alpha=alpha_fill)
    ax.plot(current_density,y_ub,color=v_color_line,lw=v_lw,ls=v_ls,label="$V_{\Omega,a-c}$")
    ax.fill_between(current_density,y_lb,y_ub,color=v_color_fill,alpha=alpha_fill)
    ax.text(x=xtext,y=y_mid,s="$V_{\Omega,a-c}$",color=v_color_line,**text_props)

    y_lb+=V_ohm_electrode
    y_ub+=V_ohm_electrolyte
    y_mid = np.max(y_lb + (y_ub-y_lb)/2)
    v_color_line = "lime"
    v_color_fill = "lawngreen"
    # ax.plot(current_density,U_rev+V_act_a+V_act_c+V_ohm_electrode+V_ohm_electrolyte,color=v_color_line,lw=v_lw,ls=v_ls,label="$V_{\Omega,KOH}$")
    # ax.fill_between(current_density,U_rev+V_act_a+V_act_c+V_ohm_electrode,U_rev+V_act_a+V_act_c+V_ohm_electrode+V_ohm_electrolyte,color=v_color_fill,alpha=alpha_fill)
    ax.plot(current_density,y_ub,color=v_color_line,lw=v_lw,ls=v_ls,label="$V_{\Omega,KOH}$")
    ax.fill_between(current_density,y_lb,y_ub,color=v_color_fill,alpha=alpha_fill)
    ax.text(x=xtext,y=y_mid,s="$V_{\Omega,KOH}$",color=v_color_line,**text_props)

    y_lb+=V_ohm_electrolyte
    y_ub+=V_ohm_membrane
    y_mid = np.max(y_lb + (y_ub-y_lb)/2)
    v_lw = 1.5
    v_ls = "solid"
    v_color_line = "darkcyan"
    v_color_fill = "aqua"
    # ax.plot(current_density,U_rev+V_act_a+V_act_c+V_ohm_electrode+V_ohm_electrolyte+V_ohm_membrane,color=v_color_line,lw=v_lw,ls=v_ls,label="$V_{\Omega,mem}$")
    # ax.fill_between(current_density,U_rev+V_act_a+V_act_c+V_ohm_electrode+V_ohm_electrolyte,U_rev+V_act_a+V_act_c+V_ohm_electrode+V_ohm_electrolyte+V_ohm_membrane,color=v_color_fill,alpha=alpha_fill)
    ax.plot(current_density,y_ub,color=v_color_line,lw=v_lw,ls=v_ls,label="$V_{\Omega,mem}$")
    ax.fill_between(current_density,y_lb,y_ub,color=v_color_fill,alpha=alpha_fill)
    ax.text(x=xtext,y=y_mid,s="$V_{\Omega,mem}$",color=v_color_line,**text_props)

    v_lw = 0.5
    ax.plot(current_density,V_cell,color="black",ls="--",label="V_{cell}")

    ax.set_xlabel("Current Density [A/cm^2]")
    x0 = ax.get_xlim()[0]
    x1 = ax.get_xlim()[1] + x0
    ax.set_xlim([0,x1])
    ax.set_ylim([0,ax.get_ylim()[1]])
    # ax.set_xlim([np.min(current_density),np.max(current_density)])
    # ax.set_ylim([0,np.max(V_cell)])
    ax.set_ylabel("Cell Voltage [V/cell]")
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__),"Alkaline_IV-Curve-{}.pdf".format(file_desc)),bbox_inches = "tight")
    plt.close()
    






