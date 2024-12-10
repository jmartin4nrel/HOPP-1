import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

def set_A_mat(A_norm,quad,inter,cubic=False):

    # Set up final A matrix for linear
    A_lin = np.vstack([
        A_norm[:,0],
        A_norm[:,1],
        A_norm[:,2],
        np.ones(np.shape(A_norm[:,0])),
        np.ones(np.shape(A_norm[:,0])),
        np.ones(np.shape(A_norm[:,0]))])

    # Set up final A matrix for quadratic without interations
    if quad:
        A_quad = np.vstack([
            np.square(A_norm[:,0]),
            np.square(A_norm[:,1]),
            np.square(A_norm[:,2]),
            A_lin])

        # Set up final A matrix for quadratic with interations
        if inter:
            A_quad_int = np.vstack([
                np.multiply(A_norm[:,0],A_norm[:,1]),
                np.multiply(A_norm[:,0],A_norm[:,2]),
                np.multiply(A_norm[:,1],A_norm[:,2]),
                A_quad])
            A = A_quad_int
        else:
            A = A_quad
    else:
        A = A_lin
    if cubic:
        A = np.vstack([
        np.multiply(np.square(A_norm[:,0]),A_norm[:,0]),
        np.multiply(np.square(A_norm[:,1]),A_norm[:,1]),
        np.multiply(np.square(A_norm[:,2]),A_norm[:,2]),
        A])
    A = np.transpose(A)

    return A

def save_corrs(doe_df, fit_var, fit_label, levels, quad, inter, var_names, var_labels, plot=False, filepath=None, log_inc=False, add_df=None):

    # Get A1, A2, and A3 as sorbent weight, hydrogenation pressure, and hydrogenation temperature
    A1 = doe_df.loc[:,var_names[0]].values
    A2 = doe_df.loc[:,var_names[1]].values
    A3 = doe_df.loc[:,var_names[2]].values

    # Set up normalized A vectors and b vector
    A_orig = np.transpose(np.vstack((A1,A2,A3)))
    A_mean = np.mean(A_orig,0,keepdims=True)
    A_std = np.std(A_orig,0,keepdims=True)
    A_norm = np.divide(np.subtract(A_orig,A_mean),A_std)
    b = doe_df.loc[:,fit_var].values
    b_mean = np.mean(b)
    b_std = np.std(b)
    b_norm = np.divide(np.subtract(b,b_mean),b_std)
    b_norm = np.transpose(np.array([b_norm,]))

    # Set full A matrix
    A = set_A_mat(A_norm,quad,inter)

    # Do linear regression
    X = np.linalg.lstsq(A,b_norm)[0]

    # Set up plots
    if var_names[1] == 'hyd_P_bar':
        x_levels = np.arange(8,33,1)
    elif var_names[1] == 'MeOH_sel':
        x_levels = np.arange(20,103,5)
    elif var_names[1] == 'h2_ratio':
        x_levels = np.arange(0.18,0.281,.01)
    if var_names[2] == 'hyd_T_C':
        y_levels = np.arange(196,256,2)
    elif var_names[2] == 'CO2_conv':
        y_levels = np.arange(56,102,2)
    elif var_names[2] == 'CO2_uptake':
        y_levels = np.arange(40,105,2)
    elif var_names[2] == 'prod_rec':
        y_levels = np.arange(50,132,4)
    elif var_names[2] == 'rec_ratio':
        y_levels = np.arange(1,5.1,0.2)
    elif var_names[2] == 'tonne_cat':
        y_levels = np.arange(1200,2801,100)
    # y_levels = np.arange(196,256,2)
    [X_grid,Y_grid] = np.meshgrid(x_levels,y_levels)
    X_vec = np.reshape(X_grid,(np.product(X_grid.shape)))
    Y_vec = np.reshape(Y_grid,(np.product(Y_grid.shape)))
    plt.set_cmap('turbo')

    # Loop through sorbent loadings
    A1s = np.unique(A1)
    num_plots = len(A1s)
    ax_list = []
    num_points = 0
    for i, wt_pct in enumerate(A1s):
        
        # Calculate fit surface at this loading
        A1 = np.ones(len(X_vec))*wt_pct
        A2 = X_vec
        A3 = Y_vec
        A = np.transpose(np.vstack((A1,A2,A3)))
        A_norm = np.divide(np.subtract(A,A_mean),A_std)
        A = set_A_mat(A_norm,quad,inter)
        fit = np.reshape((np.matmul(A,X))*b_std+b_mean,X_grid.shape)
        fit = np.minimum(fit,max(levels))
        fit = np.maximum(fit,min(levels))


        if log_inc:
            fit = 100-np.power(10,fit)

        if plot:

            if i == 0:
                ax0 = plt.subplot(2,3,i+1)
                # ax.set_position([ax.get_position().x0,ax.get_position().width,ax.get_position().y0+ax.get_position().height/2,ax.get_position().height/2])
                ax0.set_position([.1,.9,.2,.05])
                ax0.text(0.5,-4,'H2:Methanol Ratio\nCorrelations',horizontalalignment='center',fontsize=18)
            if i == 1:
                ax1 = plt.subplot(2,3,i+1)
                # ax.set_position([ax.get_position().x0,ax.get_position().width,ax.get_position().y0+ax.get_position().height/2,ax.get_position().height/2])
                ax1.set_position([.4,.75,.27,.05])
                # cplot = ax1.contourf(X_grid,Y_grid,fit,levels=levels)
                c = plt.colorbar(cax=ax1,label=fit_label,orientation='horizontal')
                c.__setattr__('labelpad',5)
            if i == 2:
                ax2 = plt.subplot(2,3,i+1)
                # ax.set_position([ax.get_position().x0,ax.get_position().width,ax.get_position().y0+ax.get_position().height/2,ax.get_position().height/2])
                ax2.set_position([.78,.9,.2,.05])
                # ax2.plot(0,0,'o',markeredgecolor='k',label='Experimental data')
                ax2.plot(0,0,'o',markeredgecolor='k',color=[1,1,1],label='Aspen models')
                ax2.plot(0,0,'o',markeredgecolor='m',color=[1,1,1],label='Corrleated data\nat DOE points\nnot run in ASPEN')
                # leg = ax2.legend(loc='lower center',bbox_to_anchor=(0.5,-4))
                leg = ax2.legend(loc='lower center',bbox_to_anchor=(0.5,-5.5))
                
            # Make a surface plot for this loading
            ax = plt.subplot(2,3,i+4)
            if i == 0:
                ax.set_position([.1,.1,.2,.45])
            if i == 1:
                ax.set_position([.44,.1,.2,.45])
            if i == 2:
                ax.set_position([.78,.1,.2,.45])
            ax_list.append(ax)
            cplot = plt.contourf(X_grid,Y_grid,fit,levels=levels)
            plt.xlabel(var_labels[1])
            plt.ylabel(var_labels[2],rotation=0)
            if var_names[2] == 'prod_rec':
                ax.yaxis.set_label_coords(-.33,0.5)
            else:
                ax.yaxis.set_label_coords(-.3,0.5)
            plt.title(var_labels[0].format(wt_pct))
            
            # # Set max and min of axes
            # xmin = np.min(X_grid)
            # xmax = np.max(X_grid)
            # ymin = np.min(Y_grid)
            # ymax = np.max(Y_grid)
            # xmin = xmin-(xmax-xmin)/10
            # xmax = xmax+(xmax-xmin)/10
            # ymin = ymin-(ymax-ymin)/10
            # ymax = ymax+(ymax-ymin)/10
            # plt.xlim((xmin,xmax))
            # plt.ylim((ymin,ymax))

            # Plot points for this loading
            cmap_min = np.min(cplot.levels)
            cmap_max = np.max(cplot.levels)
            colors = cplot.cmap.colors
            color_N = cplot.cmap.N
            points = A_orig[np.where(A_orig[:,0]==wt_pct)]
            for point_ind, point in enumerate(points):
                point_color_ind = int(np.round((doe_df.loc[point_ind+num_points,fit_var]-cmap_min)/(cmap_max-cmap_min)*color_N))-1
                if log_inc:
                    point_color_ind = int(np.round(((100-10**doe_df.loc[point_ind+num_points,fit_var])-cmap_min)/(cmap_max-cmap_min)*color_N))
                duplicate_ind = doe_df.loc[point_ind+num_points,'duplicate']
                add_in = doe_df.loc[point_ind+num_points,'doe_input'] == 0
                plt.plot([point[1]-duplicate_ind*1.,point[1]],[point[2]-duplicate_ind*2.5,point[2]],'k-',zorder=1)
                p = plt.plot(point[1]-duplicate_ind*1.,point[2]-duplicate_ind*2.5,'o',color=colors[point_color_ind],zorder=2,markeredgecolor='k')
                if add_in:
                    plt.setp(p,'markeredgecolor','m')
            num_points += len(points)

            if add_df is not None:

                add_df_inds = add_df.loc[:,'doe_input'] == 0
                add_df = add_df.iloc[np.where(add_df_inds)]
                
                Anew1 = add_df.loc[:,var_names[0]].values
                Anew2 = add_df.loc[:,var_names[1]].values
                Anew3 = add_df.loc[:,var_names[2]].values
                Anew = np.transpose(np.vstack((Anew1,Anew2,Anew3)))
                Anew_norm = np.divide(np.subtract(Anew,A_mean),A_std)
                Anew = set_A_mat(Anew_norm,quad,inter)
                bnew = np.reshape((np.matmul(Anew,X))*b_std+b_mean,add_df.shape[0])
                
                for i, point in enumerate(bnew):
                    point_color_ind = int(np.round((point-cmap_min)/(cmap_max-cmap_min)*color_N))-1
                    plt.plot(add_df[var_names[1]],add_df[var_names[2]],'o',color=colors[point_color_ind],zorder=2,markeredgecolor='m',markeredgewidth=2)

            
            

    if plot:

        # plt.gcf().set_tight_layout(True)
        plt.gcf().set_size_inches(12,5)
        plt.show()

    if filepath is not None:

        results_dict = {
            'fit_var': fit_var,
            'fit_label': fit_label,
            'levels': levels,
            'quad': quad,
            'inter': inter,
            'var_names': var_names,
            'var_labels': var_labels,
            'X': X,
            'A_mean': A_mean,
            'A_std': A_std,
            'b_mean': b_mean,
            'b_std': b_std 
        }

        results_writer = open(str(filepath)+".pkl", 'wb')
        pickle.dump(results_dict, results_writer)

if __name__ == '__main__':

    # Read in experimental data
    current_dir = Path(__file__).parent.absolute()
    
    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_adj.csv')
    
    # Choose variable to fit surface to
    fit_var = 'CO2_uptake'
    
    # fit_label = 'Methanol production [umol/g]\n\n\n'
    fit_label = 'Strong CO2 uptake [umol/g]\n\n\n'

    levels = np.arange(40,111,5)

    # Choose quadratic or linear, interactions or no
    quad = True
    inter = True

    var_names = ['sorbent_wt_pct','hyd_P_bar','hyd_T_C']
    var_labels = ['Sorbent Loading = {:.2f} wt %','Hydrogenation Pressure [bar]','Hydrog-\nenation\nTemp.\n[deg. C]']
    
    save_corrs(doe_df, fit_var, fit_label, levels, quad, inter, var_names, var_labels, log_inc=False, plot=False, filepath=current_dir/'outputs'/'co2up')

    
    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_adj.csv')
    fit_var = 'MeOH_sel'
    fit_label = 'Methanol selectivity [%]\n\n\n'
    levels = np.arange(10,101,5)
    quad = True
    inter = True
    var_names = ['sorbent_wt_pct','hyd_P_bar','hyd_T_C']
    var_labels = ['Sorbent Loading = {:.2f} wt %','Hydrogenation Pressure [bar]','Hydrog-\nenation\nTemp.\n[deg. C]']

    save_corrs(doe_df, fit_var, fit_label, levels, quad, inter, var_names, var_labels, log_inc=False, plot=False, filepath=current_dir/'outputs'/'meoh_sel')

    
    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_adj.csv')
    fit_var = 'CO2_conv'
    fit_label = 'CO2 conversion [%]\n\n\n'
    levels = np.arange(55,101,5)
    quad = True
    inter = True
    var_names = ['sorbent_wt_pct','hyd_P_bar','hyd_T_C']
    var_labels = ['Sorbent Loading = {:.2f} wt %','Hydrogenation Pressure [bar]','Hydrog-\nenation\nTemp.\n[deg. C]']

    save_corrs(doe_df, fit_var, fit_label, levels, quad, inter, var_names, var_labels, log_inc=False, plot=False, filepath=current_dir/'outputs'/'co2conv')

    # doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_out.csv')


    ''' 
    Correlations with 2 projected points missing from DOE

    '''

    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_adj.csv')
    fit_var = 'rec_ratio'
    fit_label = 'RATIO of Methanol Productivity, Recycle:Single-Pass\n[umol/g-cat:umol/g-cat]\n\n'
    levels = np.arange(1.,6.1,.5)
    quad = True
    inter = False
    var_names = ['sorbent_wt_pct','MeOH_sel','CO2_conv']
    var_labels = ['Sorbent Loading = {:.2f} wt %','Methanol Selectivity %','CO2\nConversion\nEfficiency\n[%]']

    save_corrs(doe_df, fit_var, fit_label, levels, quad, inter, var_names, var_labels, log_inc=False, plot=False, filepath=current_dir/'outputs'/'rec_ratio')


    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_adj.csv')
    fit_var = 'tonne_cat'
    fit_label = 'Catalyst Mass [tonne]\n\n\n'
    levels = np.arange(1000,3201,200)
    quad = True
    inter = False
    var_names = ['sorbent_wt_pct','hyd_P_bar','prod_rec']
    var_labels = ['Sorbent Loading = {:.2f} wt %','Hydrogenation Pressure [bar]','Methanol\nProductivity\nwith\nRecycle\n[umol/g-cat]']

    save_corrs(doe_df, fit_var, fit_label, levels, quad, inter, var_names, var_labels, log_inc=False, plot=False, filepath=current_dir/'outputs'/'tonne_cat')


    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_adj.csv')
    fit_var = 'h2_ratio'
    fit_label = 'Methanol:H2 Ratio\n\n\n'
    levels = np.arange(0.17,0.32,0.01)
    quad = True
    inter = False
    var_names = ['sorbent_wt_pct','hyd_P_bar','CO2_uptake']
    var_labels = ['Sorbent Loading = {:.2f} wt %','Hydrogenation Pressure [bar]','Strong\nCO2\nUptake\n[umol/g]']

    save_corrs(doe_df, fit_var, fit_label, levels, quad, inter, var_names, var_labels, log_inc=False, plot=False, filepath=current_dir/'outputs'/'h2_ratio')

 
    ''' 
    Final economic correlations

    '''



    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_adj.csv')
    fit_var = 'LCOM'
    fit_label = 'Levelized Cost of Methanol (LCOM) [$/kg]\n\n\n'
    levels = np.arange(0.6,1.1,0.05)
    quad = True
    inter = True
    var_names = ['sorbent_wt_pct','hyd_P_bar','hyd_T_C']
    var_labels = ['Sorbent Loading = {:.2f} wt %','Hydrogenation Pressure [bar]','Hydrog-\nenation\nTemp.\n[deg. C]']

    save_corrs(doe_df, fit_var, fit_label, levels, quad, inter, var_names, var_labels, log_inc=False, plot=True, filepath=current_dir/'outputs'/'1step_LCOM')
    
    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_adj.csv')
    add_pts = pd.read_csv(current_dir/'inputs'/'doe_inputs_adj.csv')
    fit_var = 'tonne_cat'
    fit_label = 'Catalyst Mass [tonne]\n\n\n'
    levels = np.arange(1000,3401,200)
    quad = True
    inter = True
    var_names = ['sorbent_wt_pct','hyd_P_bar','hyd_T_C']
    var_labels = ['Sorbent Loading = {:.2f} wt %','Hydrog. P. [bar]','Hydrog. T. [deg C]']

    save_corrs(doe_df, fit_var, fit_label, levels, quad, inter, var_names, var_labels, log_inc=False, plot=True, filepath=current_dir/'outputs'/'1step_tonne_cat')


    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_out.csv')
    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_adj.csv')
    fit_var = 'h2_ratio'
    fit_label = 'Methanol:H2 Ratio\n\n\n'
    levels = np.arange(0.17,0.28,0.01)
    quad = True
    inter = True
    var_names = ['sorbent_wt_pct','hyd_P_bar','hyd_T_C']
    var_labels = ['Sorbent Loading = {:.2f} wt %','Hydrog. P. [bar]','Hydrog. T. [deg C]']

    save_corrs(doe_df, fit_var, fit_label, levels, quad, inter, var_names, var_labels, log_inc=False, plot=True, filepath=current_dir/'outputs'/'1steph2_ratio')


    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_adj.csv')
    # add_pts = pd.read_csv(current_dir/'inputs'/'doe_inputs_adj.csv')
    fit_var = 'LCOM'
    fit_label = 'Levelized Cost of Methanol (LCOM) [$/kg]\n\n\n'
    levels = np.arange(0.6,1.1,0.05)
    quad = True
    inter = False
    var_names = ['sorbent_wt_pct','h2_ratio','tonne_cat']
    var_labels = ['Sorbent Loading = {:.2f} wt %','H2:Methanol Ratio','Catalyst\nMass\n[Tonne]']

    save_corrs(doe_df, fit_var, fit_label, levels, quad, inter, var_names, var_labels, log_inc=False, plot=True, filepath=current_dir/'outputs'/'LCOM')#, add_df=add_pts)
    
    
    
    pickle_reader = open(current_dir/'outputs'/'co2up.pkl','rb')
    old_results = pickle.load(pickle_reader)

    fit_var = old_results['fit_var']
    fit_label = old_results['fit_label']
    levels = old_results['levels']
    quad = old_results['quad']
    inter = old_results['inter']
    var_names = old_results['var_names']
    var_labels = old_results['var_labels']
    X = old_results['X']
    A_mean = old_results['A_mean']
    A_std = old_results['A_std']
    b_mean = old_results['b_mean']
    b_std = old_results['b_std']
    
    save_corrs(doe_df, fit_var, fit_label, levels, quad, inter, var_names, var_labels, plot=False)

