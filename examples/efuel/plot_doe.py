import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def set_A_mat(A_norm,quad,inter):

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
    A = np.transpose(A)

    return A

if __name__ == '__main__':

    # Read in experimental data
    current_dir = Path(__file__).parent.absolute()
    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs.csv')
    doe_df = pd.read_csv(current_dir/'inputs'/'doe_inputs_out.csv')

    # Choose variable to fit surface to
    
    # fit_var = 'MeOH_prod'
    # fit_var = 'MeOH_sel'
    # fit_var = 'CO2_uptake'
    fit_var = 'h2_ratio'
    
    # fit_label = 'Methanol production [umol/g]\n\n\n'
    # fit_label = 'Methanol selectivity [%]\n\n\n'
    # fit_label = 'Strong CO2 uptake [umol/g]\n\n\n'
    fit_label = 'Hydrogen:Methanol Ratio [kg/kg]'

    # levels = np.arange(10,61,5)
    # levels = np.arange(10,101,10)
    # levels = np.arange(40,140,10)
    levels = np.arange(0.18,0.28,0.01)

    b = doe_df.loc[:,fit_var].values

    # Choose quadratic or linear, interactions or no
    quad = True
    inter = False#True

    # Get A1, A2, and A3 as sorbent weight, hydrogenation pressure, and hydrogenation temperature
    A1 = doe_df.loc[:,'sorbent_wt_pct'].values
    A2 = doe_df.loc[:,'MeOH_prod'].values
    A3 = doe_df.loc[:,'CO2_uptake'].values

    # Set up normalized A vectors and b vector
    A_orig = np.transpose(np.vstack((A1,A2,A3)))
    A_mean = np.mean(A_orig,0,keepdims=True)
    A_std = np.std(A_orig,0,keepdims=True)
    A_norm = np.divide(np.subtract(A_orig,A_mean),A_std)
    b_mean = np.mean(b)
    b_std = np.std(b)
    b_norm = np.divide(np.subtract(b,b_mean),b_std)
    b_norm = np.transpose(np.array([b_norm,]))

    # Set full A matrix
    A = set_A_mat(A_norm,quad,inter)

    # Do linear regression
    X = np.linalg.lstsq(A,b_norm)[0]

    # Set up plots
    hyd_P = np.arange(8,33,1)
    hyd_T = np.arange(196,256,2)
    [X_grid,Y_grid] = np.meshgrid(hyd_P,hyd_T)
    X_vec = np.reshape(X_grid,(np.product(X_grid.shape)))
    Y_vec = np.reshape(Y_grid,(np.product(Y_grid.shape)))
    plt.set_cmap('turbo')

    # Loop through sorbent loadings
    num_plots = 3
    sorb_wt_pcts = np.unique(A1)
    ax_list = []
    num_points = 0
    for i, wt_pct in enumerate(sorb_wt_pcts):
        
        # Calculate fit surface at this loading
        A1 = np.ones(len(X_vec))*wt_pct
        A2 = X_vec
        A3 = Y_vec
        A = np.transpose(np.vstack((A1,A2,A3)))
        A_norm = np.divide(np.subtract(A,A_mean),A_std)
        A = set_A_mat(A_norm,quad,inter)
        fit = np.reshape((np.matmul(A,X))*b_std+b_mean,X_grid.shape)

        # Make a surface plot for this loading
        ax = plt.subplot(1,3,i+1)
        ax_list.append(ax)
        cplot = plt.contourf(X_grid,Y_grid,fit,levels=levels)
        plt.colorbar(label=fit_label,location='right')
        plt.xlabel('Hydrog. P. [bar]')
        plt.ylabel('Hydrog. T. [deg C]')
        plt.title('Sorbent Loading = {:.2f} wt %'.format(wt_pct))
        
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
            point_color_ind = int(np.round((doe_df.loc[point_ind+num_points,fit_var]-cmap_min)/(cmap_max-cmap_min)*color_N))
            duplicate_ind = doe_df.loc[point_ind+num_points,'duplicate']
            plt.plot([point[1]-duplicate_ind*1.,point[1]],[point[2]-duplicate_ind*2.5,point[2]],'k-',zorder=1)
            plt.plot(point[1]-duplicate_ind*1.,point[2]-duplicate_ind*2.5,'o',color=colors[point_color_ind],zorder=2,markeredgecolor='k')
        num_points += len(points)

    plt.gcf().set_tight_layout(True)
    plt.gcf().set_size_inches(12,3)
    plt.show()