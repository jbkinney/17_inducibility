import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import re


def thermodynamic_model(t_sat, t_bg, P, F, alpha, beta=1):
    """ Returns predictions of a thermodynamic model"""
    return (t_sat*P + t_sat*P*alpha*beta*F)/(1 + P + F + alpha*F*P) + t_bg

# Function for plotting thermodynamic model
def plot_manifold_model(ax, t_sat, t_bg, F, alpha, beta=1, 
                           color='k', 
                           linewidth=1, 
                           P_min=1E-8, 
                           P_max=1E3, 
                           num_P=1000,
                           label=None,
                           opacity=1,
                           lim=None):
    
    # Plot model
    Ps = np.logspace(np.log10(P_min), np.log10(P_max), 100)
    model_xs = thermodynamic_model(t_sat=t_sat,
                                   t_bg=t_bg,
                                   alpha=alpha,
                                   beta=beta,
                                   F=0,
                                   P=Ps)
    model_ys = thermodynamic_model(t_sat=t_sat,
                                   t_bg=t_bg,
                                   alpha=alpha,
                                   beta=beta,
                                   F=F,
                                   P=Ps)
    ax.loglog(model_xs, model_ys, '-', color=color, alpha=opacity, label=label)
    if lim is not None:
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    


def get_measurement_subset_df(data_df, regex):
    prog = re.compile(regex)
    rows = [bool(prog.match(name)) for name in data_df.index]
    tmp_df = data_df\
                .loc[rows, ['log_t+', 'dlog_t+','log_t-','dlog_t-', 'outlier']]\
                .copy()
    tmp_df.dropna(inplace=True)
    tmp_df['outlier'] = tmp_df['outlier'].astype(bool)
    return tmp_df

# Function for plotting measurements 
def plot_manifold_measurements(ax,
                               df,
                               samples_labels_colors, 
                               fontsize=12, 
                               ticks=[1E-3,1E-2,1E-1,1E0,1E1],
                               markersize=4,
                               lim=[5E-4,5E1],
                               ylabel="$t_+$ (a.u.) ",
                               xlabel="$t_-$ (a.u.) ",
                               show_legend=True,
                               legend_loc='lower right',
                               show_outliers=True,
                               show_errorbars=False,
                               show_dots=True):
    
    # Set plotting parameters
    lim = np.array(lim)

    # Draw diagonal
    ax.loglog(lim, lim,':',color='k',alpha=0.5)
    
    # Draw data points
    num_points = 0
    num_outliers = 0
    for sample, label, color in samples_labels_colors:

        # Extract data points for sample
        tmp_df = get_measurement_subset_df(df, sample)

        # Compute the number of datapoints
        num_points += len(tmp_df)
        print('%s: %d points'%(sample, len(tmp_df)))
        #pdb.set_trace()
     
        tmp_df.dropna(inplace=True)
        if len(tmp_df)>=1:

            xs = np.exp(tmp_df['log_t-'].values)
            ys = np.exp(tmp_df['log_t+'].values)

            if show_dots:
                ax.loglog(xs,ys,'o',label=label, color=color, 
                    markersize=markersize)

            dlog_xs = tmp_df['dlog_t-'].values
            dlog_ys = tmp_df['dlog_t+'].values

            xs_hi = xs * np.exp(1.96*dlog_xs)
            xs_lo = xs * np.exp(-1.96*dlog_xs)

            ys_hi = ys * np.exp(1.96*dlog_ys)
            ys_lo = ys * np.exp(-1.96*dlog_ys)

            if show_errorbars:
                K = len(xs)
                for k in range(K):
                    x = xs[k]
                    x_hi = xs_hi[k]
                    x_lo = xs_lo[k]
                    y = ys[k]
                    y_hi = ys_hi[k]
                    y_lo = ys_lo[k]

                    plt.loglog([x_hi, x_lo], [y, y], color='k', linewidth=1)
                    plt.loglog([x, x], [y_hi, y_lo], color='k', linewidth=1)

            # Mark outliers: We need to change how this is done
            if show_outliers:
                os = tmp_df['outlier'].values
                num_outliers += sum(os)
                ax.loglog(xs[os],ys[os], 'kx', 
                  alpha=.5,
                  markersize=1.5*markersize)

    # Adjust plot appearance
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.get_xaxis().set_tick_params(which='minor', size=0, labelsize=fontsize) 
    ax.get_yaxis().set_tick_params(which='minor', size=0, labelsize=fontsize)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if show_legend:
        ax.legend(loc=legend_loc, handletextpad=0, fontsize=fontsize)

    samples = ' + '.join([s for s, l, c in samples_labels_colors])
    print('n = %d-%d points plotted for  %s' % 
        (num_points, num_outliers, samples))

    return num_points, num_outliers

# Load all measurements
def get_clonal_measurements_df():
    # data_file = '../intermediate/clonal_measurements.txt'
    # data_df = pd.read_csv(data_file, sep='\t', comment='#')
    # data_df.set_index('name', inplace=True)
    #file_name = '../data/measurements_summary.xlsx'
    file_name = '../data/results.xlsx'
    sheet_name = 'measurements_summary'
    data_df = pd.read_excel(file_name, sheet_name=sheet_name)
    data_df.set_index('name',inplace=True)
    return data_df


# Load c61 model parameters
def get_resampled_params_df(architecture):
    files_dict = {
        'c61':'../intermediate/resampled_params_for_c61.txt',
        'c71':'../intermediate/resampled_params_for_c71.txt',
        'occlusion':'../intermediate/resampled_params_for_occlusion.txt'
    }
    assert architecture in files_dict, \
        'architecture = %s;  must be one of %s' %\
        (architecture, list(files_dict.keys()))

    param_file = files_dict[architecture]
    param_df = pd.read_csv(param_file, delim_whitespace=True, comment='#')
    param_df.set_index('run', inplace=True, drop=True)
    return param_df


def get_distance_params_df():
    distance_file = '../intermediate/params_versus_distance.txt'
    distance_df = pd.read_csv(distance_file, 
        delim_whitespace=True, comment='#')
    distance_df.set_index('distance', inplace=True)
    return distance_df



