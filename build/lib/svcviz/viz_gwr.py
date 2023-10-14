import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely import wkt
from .utils import *

"""
Author Information:
    - Name: Victor Irekponor, Taylor Oshan
    - Email: vireks@umd.edu, toshan@umd.edu
    - Date Created: 2023-10-12
"""


def compare_surfaces_grid(data, vars, use_tvalues=True, savefig=None):

    """
    Create a grid of surface plots for a set of variables from a dataset. Each plot shows one variable,
    and t-values can be overlaid to highlight significant regions. The colormap is adjusted based on the 
    data values to make visualization more intuitive.
    ================================
    Parameters:
    - data (DataFrame or similar): The dataset containing the variables and t-values to plot.
    - vars (list of str): The names of the variables to plot. Each variable should correspond to a column in 'data'.
    - use_tvalues (bool): Whether to overlay t-values on the plots to highlight significant regions. Defaults to True.
    - savefig (str or None): The file path to save the figure to. If None, the figure is not saved. Defaults to None.

    Returns:
    - None: Displays a grid of surface plots.
    """
    n_vars = len(vars)
    tvalues = ['t_' + var for var in vars]  # Automatically generate tvalue column names

    grid_dim = int(np.ceil(np.sqrt(n_vars)))
    
    # Adjusting the figsize based on number of variables
    if n_vars in [1, 2]:
        figsize = (8, 4 * n_vars)  # Adjusting the height for 1 variable
        fig, axes = plt.subplots(nrows=n_vars, ncols=1, figsize=figsize)
    else:
        figsize = (20, 20)
        fig, axes = plt.subplots(nrows=grid_dim, ncols=grid_dim, figsize=figsize)
        
    
    if n_vars == 1 :
        axes = [axes]
    else:
        axes = axes.ravel()

    cmap = plt.cm.seismic
    vmin = min(data[var].min() for var in vars)
    vmax = max(data[var].max() for var in vars)
    if (vmin < 0) & (vmax < 0):
        cmap = truncate_colormap(cmap, 0.0, 0.5)
    elif (vmin > 0) & (vmax > 0):
        cmap = truncate_colormap(cmap, 0.5, 1.0)
    else:
        cmap = shift_colormap(cmap, start=0.0,
                              midpoint=1 - vmax / (vmax + abs(vmin)), stop=1.)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    for i, var in enumerate(vars):
        ax = axes[i]
        ax.set_title(var, fontsize=15)
        data.plot(var, cmap=sm.cmap, ax=ax, vmin=vmin, vmax=vmax, edgecolor='k', linewidth=0.5)
        if use_tvalues:
            tvalue_col = tvalues[i]
            data[data[tvalue_col] == 0].plot(color='lightgrey', edgecolor='white', ax=ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if n_vars > 2:
        for j in range(i+1, grid_dim*grid_dim):
            axes[j].axis('off')
            
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.04, hspace=0)
    cax = fig.add_axes([1.0, 0.1, 0.03, 0.8])
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()

def viz_gwr(col_names, df_geo, gwr_object, use_tvalues=True,  alpha=0.05, coef_surfaces=None):
    """
    Visualize Geographically Weighted Regression (GWR) results by plotting coefficient surfaces
    and optionally overlaying t-values to highlight significant regions.
    ================================
    Parameters:
    - col_names (list of str): The names of the coefficients (excluding intercept and geometry).
    - df_geo (Geometry): The geometry column of the main dataframe
    - gwr_object (GWR result object): The object containing GWR results including coefficients and t-values.
    - use_tvalues (bool): Whether to overlay t-values on the coefficient surfaces. Defaults to True.
    - alpha (float): The significance level for filtering t-values. Defaults to 0.05.
    - coef_surfaces (list of str or None): Specific coefficients to plot, if None, all coefficients are plotted. Defaults to None.

    Returns:
    - None: Displays the coefficient surfaces plots.
    """

    data = gpd.GeoDataFrame(gwr_object.params, geometry=df_geo)
    col_names = ['intercept'] + col_names + ['geometry']
    data.columns = col_names

    tvl = pd.DataFrame(gwr_object.filter_tvals(alpha=alpha))
    tvl.columns = ['t_'+col for col in col_names if col !='geometry']
    merged = data.merge(tvl, left_index=True, right_index=True)

    col_names.pop()
    
    tval_names = [col for col in merged.columns if col.startswith('t')]
    if coef_surfaces is not None:
        compare_surfaces_grid(merged, coef_surfaces, use_tvalues=use_tvalues)
    else:
        compare_surfaces_grid(merged, col_names, use_tvalues=use_tvalues)
        

def viz_gw(df_geo, betas, std_errs, use_tvalues=True, coef_surfaces=None, alpha=0.05):
    """
    Visualize Geographically Weighted (GW) results by plotting coefficient surfaces
    and optionally overlaying t-values to highlight significant regions.
    ================================
    Parameters:
    - df_geo (Geometry): The geometry column of the main dataframe
    - betas (DataFrame): The DataFrame containing beta coefficients.
    - std_errs (DataFrame): The DataFrame containing standard errors corresponding to beta coefficients.
    - use_tvalues (bool): Whether to overlay t-values on the coefficient surfaces. Defaults to True.
    - coef_surfaces (list of str or None): Specific coefficients to plot, if None, all coefficients are plotted. Defaults to None.
    - alpha (float): The significance level for filtering t-values. Defaults to 0.05.

    Returns:
    - None: Displays the coefficient surfaces plots with optional t-values overlaid.
    """

    # data = gpd.GeoDataFrame(betas, geometry=df_geo) # throw exception if columns are not named
    betas.columns = ['beta_'+col for col in betas.columns]
    
    std_errs.columns = ['std_'+std for std in std_errs.columns]
    
    data = merge_index(betas, std_errs)  # merge beta+std_errors
    mask = mask_insignificant_t_values(data.copy(), alpha=alpha)
    tvals = mask[[col for col in mask.columns if col.startswith('t')]]
    data_df = gpd.GeoDataFrame(merge_index(data, tvals), geometry=df_geo)
    betas = betas.columns 
    col_tvals = [col for col in tvals]
    if coef_surfaces is not None:
        compare_surfaces_grid(data_df, coef_surfaces, use_tvalues=use_tvalues)
    else:
        compare_surfaces_grid(data_df, betas, use_tvalues=use_tvalues)
    

def compare_conf(df_geo, est1, stderr1, est2, stderr2, var1,
                     var2, z_value=1.96):
    """
    Compare the confidence intervals of two estimated models, highlighting areas where the intervals overlap.
    ================================
    Parameters:
    - df_geo (Geometry): The geometry column of the main dataframe
    - est1 (DataFrame): The DataFrame containing beta coefficients of the first model.
    - stderr1 (DataFrame): The DataFrame containing standard errors corresponding to beta coefficients of the first model.
    - est2 (DataFrame): The DataFrame containing beta coefficients of the second model.
    - stderr2 (DataFrame): The DataFrame containing standard errors corresponding to beta coefficients of the second model.
    - var1 (str): The specific variable of interest from the first model to be compared.
    - var2 (str): The specific variable of interest from the second model to be compared.
    - z_value (float): The z-value for calculating the confidence interval. Defaults to 1.96 for a 95% confidence interval.

    Returns:
    - None: Displays a plot highlighting areas with overlapping confidence intervals.
    """
    
    est1.columns = ['beta_'+col if not col.startswith('beta_') else col for col in est1.columns]
    stderr1.columns = ['std_'+col if not col.startswith('std') else col for col in stderr1.columns]
    model_1 = merge_index(est1, stderr1)
    
    est2.columns = ['beta_'+col for col in est2.columns]
    stderr2.columns = ['std_'+col for col in est2.columns]
    data = merge_index(est2, stderr2)  
    
    data_df = gpd.GeoDataFrame(merge_index(model_1, data), geometry=df_geo)
        
    model_1['lower_'+var1] = model_1['beta_'+var1] - z_value * model_1['std_'+var1]
    model_1['upper_'+var1]  = model_1['beta_'+var1] + z_value * model_1['std_'+var1]
    
    data['lower_'+var2] = data['beta_'+var2] - z_value * data['std_beta_'+var2]
    data['upper_'+var2] = data['beta_'+var2] + z_value * data['std_beta_'+var2]

    data_df[var1] = ((model_1['lower_'+var1] <= data['upper_'+var2]) &
                     (model_1['upper_'+var1] >= data['lower_'+var2]))
    
    fig, ax = plt.subplots(figsize=(15, 10))
    data_df[data_df[var1]].plot(ax=ax, color='yellow', edgecolor='blue', linewidth=.3, label='Overlap')
    data_df[~data_df[var1]].plot(ax=ax, color='white', edgecolor='black', label='Overlap')
    
    ax.set_title(f'{round(data_df[var1].sum()/len(data_df)*100, 2)}% of the confidence intervals of both models overlap. {round(100-(data_df[var1].sum()/len(data_df)*100), 2)}% do not', fontsize=16)



def compare_surfaces(data, var1, var2, var1_t, var2_t, use_tvalues=False, savefig=None):
    '''
    Function that creates comparative visualization of GWR and MGWR surfaces.
    Parameters
    ----------
    data   : pandas or geopandas Dataframe
             gwr/mgwr results
    var1   : string
             name of gwr parameter estimate column in frame
    var2   : string
             name of mgwr parameter estimate column in frame
    gwr_t  : string
             name of gwr t-values column in frame associated with var1
    gwr_bw : float
             bandwidth for gwr model for var1
    mgwr_t : string
             name of mgwr t-values column in frame associated with var2
    mgwr_bw: float
             bandwidth for mgwr model for var2
    name   : string
             common variable name to use for title
    kwargs1:
             additional plotting arguments for gwr surface
    kwargs2:
             additional plotting arguments for mgwr surface
    savefig: string, optional
             path to save the figure. Default is None. Not to save figure.
    '''
    import matplotlib.pyplot as plt
    import geopandas as gp
    

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(45, 38))
    ax0 = axes[0]
#     ax0.set_title('Scenario '+ name + ' ' + data[var1].name.upper(), fontsize=40)
    ax0.set_title(var1, fontsize=50)
    ax1 = axes[1]
#     ax1.set_title('Scenario '+ name + ' ' + data[var2].name.split('_')[0].upper(), fontsize=40)
    ax1.set_title(var2, fontsize=50)

    #Set color map
    cmap = plt.cm.seismic

    #Find min and max values of the two combined datasets
    improved_min = data[var1].min()
    improved_max = data[var1].max()
    classic_min = data[var2].min()
    classic_max = data[var2].max()
    vmin = np.min([improved_min, classic_min])
    vmax = np.max([improved_max, classic_max])
    #If all values are negative use the negative half of the colormap
    if (vmin < 0) & (vmax < 0):
        cmap = truncate_colormap(cmap, 0.0, 0.5)
    #If all values are positive use the positive half of the colormap
    elif (vmin > 0) & (vmax > 0):
        cmap = truncate_colormap(cmap, 0.5, 1.0)
    #Otherwise, there are positive and negative values so the colormap so zero is the midpoint
    else:
        cmap = shift_colormap(cmap, start=0.0,
                              midpoint=1 - vmax / (vmax + abs(vmin)), stop=1.)

    #Create scalar mappable for colorbar and stretch colormap across range of data values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=vmin, vmax=vmax))

    #Plot GWR parameters
    data.plot(var1, cmap=sm.cmap, ax=ax0, vmin=vmin, vmax=vmax, edgecolor='k', linewidth=1.3)
    if (data[var1_t]==0).any() and use_tvalues==True:
        print('length is:', len(data[data[var1_t]==0]))
        data[data[var1_t]==0].plot(color='lightgrey', edgecolor='white', ax=ax0)

    #Plot MGWR parameters
    data.plot(var2, cmap=sm.cmap, ax=ax1, vmin=vmin, vmax=vmax, edgecolor='k', linewidth=1.3)
    if (data[var2_t]==0).any() and use_tvalues==True:
        data[data[var2_t]==0].plot(color='lightgrey', edgecolor='white', ax=ax1)

    #Set figure options and plot
#     fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=50)
    ax0.get_xaxis().set_visible(False)
    ax0.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()


def compare_two_surf(df_geo, est1, stderr1, est2, stderr2, var1,
                     var2, use_tvalues=False, alpha=0.05):
    """
    Compare the surfaces of two estimated models for specific variables,
    with an option to overlay t-values to highlight significant regions.
    ===========================
    Parameters:
    - df_geo (Geometry): The geometry column of the main dataframe.
    - est1 (DataFrame): The DataFrame containing beta coefficients of the first model.
    - stderr1 (DataFrame): The DataFrame containing standard errors corresponding to beta coefficients of the first model.
    - est2 (DataFrame): The DataFrame containing beta coefficients of the second model.
    - stderr2 (DataFrame): The DataFrame containing standard errors corresponding to beta coefficients of the second model.
    - var1 (str): The specific variable of interest from the first model to be compared.
    - var2 (str): The specific variable of interest from the second model to be compared.
    - use_tvalues (bool): Whether to overlay t-values on the surfaces to highlight significant regions. Defaults to False.
    - alpha (float): The significance level for filtering t-values. Defaults to 0.05.

    Returns:
    - None: Displays the comparative surfaces plots with optional t-values overlaid.
    """

    est1.columns = ['beta_'+col if not col.startswith('beta_') else col for col in est1.columns]
    stderr1.columns = ['std_'+col if not col.startswith('std') else col for col in stderr1.columns]
    model_1 = merge_index(est1, stderr1)
    model_1mask = mask_insignificant_t_values(model_1.copy(), alpha=alpha)
    tvals = model_1mask[[col for col in model_1mask.columns if col.startswith('t')]]
    model_1df = merge_index(model_1, tvals)
    
    
    est2.columns = ['beta_'+col for col in est2.columns]
    stderr2.columns = ['std_'+col for col in stderr2.columns]
    data = merge_index(est2, stderr2)  
    model2_mask = mask_insignificant_t_values(data.copy(), alpha=alpha)
    model2_tvals = model2_mask[[col for col in model2_mask.columns if col.startswith('t')]]
    
    model_2 = merge_index(data, model2_tvals)
    data_df = gpd.GeoDataFrame(merge_index(model_1df, model_2), geometry=df_geo)

    t_var1 = 't_'+var1
    t_var2 = 't_'+var2

    compare_surfaces(data_df, var1, var2, t_var1, 
                     t_var2, use_tvalues=use_tvalues)
