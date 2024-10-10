from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_sync_electrolyzer_clusters import PEM_Clusters 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


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
def get_efficiency_curve(pem: PEM_Clusters,file_desc = "test",save_file = True):
    filepath = os.path.join(os.path.dirname(__file__),"PEM_Efficiency-Curve-{}.csv".format(file_desc))
    dA = 10
    current_range = np.arange(dA,pem.nominal_current+dA,dA) 
    current_density = np.zeros(len(current_range))
    V_cell = np.zeros(len(current_range))
    H2_cell = np.zeros(len(current_range)) #kg/hr

    for ii,I_stack in enumerate(current_range):
        current_density[ii] = pem.calc_current_density(pem.T_stack, I_stack)
        V_cell[ii] = pem.cell_design(pem.T_stack,I_stack)
        H2_cell[ii] = pem.cell_H2_production_rate(pem.T_stack,I_stack)
    H2_stack = H2_cell*pem.n_cells
    Stack_Power_kW = current_range*V_cell*pem.n_cells/1e3
    power_usage_kWh = current_range*V_cell/1e3
    efficiency_kWh_pr_kg = power_usage_kWh/H2_cell
    keys = ["I [A]","V_cell [V]","Stack Power [kW]","Stack H2 Production [kg/hr]","Efficiency [kWh/kg]","Load %"]
    vals = [current_range,V_cell,Stack_Power_kW,H2_stack,efficiency_kWh_pr_kg,100*Stack_Power_kW/pem.stack_rating_kW]
    df = pd.DataFrame(dict(zip(keys,vals)))
    if save_file:
        df.to_csv(filepath)
        print("Saved efficiency curve to: {}".format(filepath))
    return df
def plot_IV_curve(pem:PEM_Clusters,file_desc = "test",save_fig = True):
    dA = 10
    # current_range = np.arange(pem.min_current,pem.nominal_current+dA,dA) 
    current_range = np.arange(dA,pem.nominal_current+dA,dA) 
    current_density = np.zeros(len(current_range))
    V_cell = np.zeros(len(current_range))

    U_rev = np.zeros(len(current_range))
    V_ohm = np.zeros(len(current_range))
    
    V_act_a = np.zeros(len(current_range))
    V_act_c = np.zeros(len(current_range))

    V_ohm_electrode = np.zeros(len(current_range))
    V_ohm_membrane = np.zeros(len(current_range))


    for ii,I_stack in enumerate(current_range):
        j = pem.calc_current_density(I_stack)
        V_cell[ii] = pem.cell_design(pem.T_stack,I_stack)
        
        current_density[ii] = j
        U_rev[ii] = pem.cell_reversible_overpotential(pem.T_stack)
        V_ohm[ii] = pem.cell_ohmic_overpotential(pem.T_stack, I_stack)
        V_act_a[ii], V_act_c[ii] = pem.cell_activation_overpotential(pem.T_stack, I_stack)
        

        R_elec = pem.cell_electrode_resistance()
        R_membrane = pem.cell_membrane_resistance(pem.T_stack)  # [Ohms] 

        V_ohm_electrode[ii] = j*R_elec
        V_ohm_membrane[ii] = j*R_membrane

    keys = ["I [A]","J [A/cm^2]","V_cell","U_rev","V_act,a","V_act,c","V_ohm,a-c","V_ohm,mem"]
    vals = [current_range,current_density,V_cell,U_rev,V_act_a,V_act_c,V_ohm_electrode,V_ohm_membrane]
    df = pd.DataFrame(dict(zip(keys,vals)))
    df.to_csv(os.path.join(os.path.dirname(__file__),"PEM_IV-Curve_Data-{}.csv".format(file_desc)))

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
    ax.plot(current_density,y_ub,color=v_color_line,lw=v_lw,ls=v_ls,label="$V_{\Omega,a-c}$")
    ax.fill_between(current_density,y_lb,y_ub,color=v_color_fill,alpha=alpha_fill)
    ax.text(x=xtext,y=y_mid,s="$V_{\Omega,a-c}$",color=v_color_line,**text_props)

    y_lb+=V_ohm_electrode
    y_ub+=V_ohm_membrane
    y_mid = np.max(y_lb + (y_ub-y_lb)/2)
    v_color_line = "lime"
    v_color_fill = "lawngreen"
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
    ax.set_ylabel("Cell Voltage [V/cell]")
    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(os.path.dirname(__file__),"PEM_IV-Curve-{}.pdf".format(file_desc)),bbox_inches = "tight")
    plt.close()
    






