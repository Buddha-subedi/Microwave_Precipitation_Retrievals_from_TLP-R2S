# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:51:12 2024

@author: subed042
"""

def plot_confusion_matrix(y_test, y_test_pred, xlabel, ylabel, title):
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm_ocean_im = confusion_matrix(y_test, y_test_pred)

    cm_tot_ocean_im = np.vstack([cm_ocean_im, np.sum(cm_ocean_im, axis=0)])
    cm_tot_ocean_im = np.hstack([cm_tot_ocean_im, np.sum(cm_tot_ocean_im, axis=1).reshape(-1, 1)])
    total_samples = np.sum(cm_ocean_im)  
    percent_equivalence = cm_ocean_im / total_samples * 100

    annotations = []

    for i in range(cm_tot_ocean_im.shape[0]):
        row_annotations = []
        for j in range(cm_tot_ocean_im.shape[1]):
            value = cm_tot_ocean_im[i, j]
            if i < cm_ocean_im.shape[0] and j < cm_ocean_im.shape[1]:
                percent_value = percent_equivalence[i, j]
                row_annotations.append(f"{value}\n({percent_value:.2f}%)")
            elif i == cm_ocean_im.shape[0] and j < cm_ocean_im.shape[1]:  # Total row
                actual_total = cm_tot_ocean_im[i, j]
                correct = cm_ocean_im[j, j]
                correct_percent = correct / actual_total * 100 if actual_total != 0 else 0
                false_percent = 100 - correct_percent
                row_annotations.append(f"{value}\n({correct_percent:.2f}% )\n({false_percent:.2f}%)")
            elif i < cm_ocean_im.shape[0] and j == cm_ocean_im.shape[1]:  # Total column
                predicted_total = cm_tot_ocean_im[i, j]
                correct = cm_ocean_im[i, i]
                correct_percent = correct / predicted_total * 100 if predicted_total != 0 else 0
                false_percent = 100 - correct_percent
                row_annotations.append(f"{value}\n({correct_percent:.2f}%)\n({false_percent:.2f}% )")
            else:  
                correct_total = np.trace(cm_ocean_im)
                correct_percent = correct_total / value * 100 if value != 0 else 0
                false_percent = 100 - correct_percent
                row_annotations.append(f"{value}\n({correct_percent:.2f}%)\n({false_percent:.2f}%)")
        annotations.append(row_annotations)


    annotations = np.array(annotations)

    mask = np.full(cm_tot_ocean_im.shape, '', dtype=object)

    for i in range(cm_ocean_im.shape[0]):
        mask[i, i] = 'diag'

    mask[-1, :] = 'total'
    mask[:, -1] = 'total'

    mask[mask == ''] = 'off_diag'

    colors = {'diag': 'lightblue', 'total': 'whitesmoke', 'off_diag': 'salmon'}
    cmap = sns.color_palette([colors[key] for key in ['diag', 'total', 'off_diag']])

    mask_num = np.zeros_like(mask, dtype=float)
    mask_num[mask == 'diag'] = 0
    mask_num[mask == 'total'] = 1
    mask_num[mask == 'off_diag'] = 2


    plt.figure(figsize=(4, 4))
    sns.heatmap(mask_num, annot=False, cmap=plt.cm.colors.ListedColormap(cmap),
                xticklabels=['noPrecip', 'Rain', 'Snow', 'Total'],
                yticklabels=['noPrecip', 'Rain', 'Snow', 'Total'],
                linewidths=1, linecolor='black', cbar=False)  

    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.tick_params(axis='both', which='both', width=2, length=6)
    for i in range(cm_tot_ocean_im.shape[0]):
        for j in range(cm_tot_ocean_im.shape[1]):
            text = annotations[i, j]
            if i == cm_ocean_im.shape[0] or j == cm_ocean_im.shape[1]: 
                parts = text.split('\n')
                plt.text(j + 0.5, i + 0.45, parts[1],
                         ha='center', va='center', fontsize=8, fontweight='bold', rotation=0, color='blue')
                plt.text(j + 0.5, i + 0.65, parts[2],
                         ha='center', va='center', fontsize=8, fontweight='bold', rotation=0, color='salmon')
            else:
                plt.text(j + 0.5, i + 0.5, text,
                         ha='center', va='center', fontsize=8, fontweight='bold', rotation=45)

    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.show()
    









def compute_cdf_and_metrics(y, x, y2):
    import scipy.stats as stats
    from scipy.interpolate import interp1d
    import numpy as np
    import pandas as pd
    from scipy.stats import gaussian_kde
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Convert input data to Pandas Series
    y_series = pd.Series(y)
    x_lrn = pd.Series(x)
    y_series = y_series.clip(lower=x_lrn.min())
    

    # Compute empirical CDF of the estimated rate (y_series)
    cumfreq_result = stats.cumfreq(y_series, numbins=len(y_series))
    Fi1 = cumfreq_result.cumcount
    xi1 = cumfreq_result.lowerlimit + np.arange(len(Fi1)) * cumfreq_result.binsize
    Fi1 = Fi1 / max(Fi1)  # Normalize to get CDF

    # Compute empirical inverse CDF of the predicted estimate rate (x_lrn)
    cumfreq_result_2 = stats.cumfreq(x_lrn, numbins=len(x_lrn))
    Fi2 = cumfreq_result_2.cumcount
    xi2 = cumfreq_result_2.lowerlimit + np.arange(len(Fi2)) * cumfreq_result_2.binsize
    Fi2 = Fi2 / max(Fi2)  # Normalize to get CDF

    
    # Compute interpolated CDF for the estimated data
    # Actually cdf_x is a set of probabilities at which you want to find the bias corrected estimated value
    cdf_x = interp1d(xi1, Fi1, kind='linear', fill_value="extrapolate")(y_series)
    
    # Map the estimated data into the observed rain rate of DPR
    x_cdf = interp1d(Fi2, xi2, kind='linear', fill_value="extrapolate")(cdf_x)
    x_cdf = np.nan_to_num(x_cdf, nan=0.01) 

    # Calculate metrics
    bias = np.mean(y2 - x_cdf)
    RMSE = np.sqrt(np.mean((x_cdf - y2) ** 2))
    MAE = np.mean(np.abs(x_cdf - y2))
    corr = np.corrcoef(x_cdf, y2)[0, 1]
    R2 = corr ** 2

    return x_cdf, bias, RMSE, MAE, corr, np.column_stack((xi2, Fi2)), np.column_stack((xi1, Fi1))




def plot_density_with_metrics_snow(x, y, x_axis_title, y_axis_title, plot_title):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Calculate metrics using the original data
    bias = np.mean(y - x)
    mae = mean_absolute_error(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    corr = np.corrcoef(x, y)[0, 1]

    # Filter out points where either x or y is less than 0.02 for plotting
    mask = (x >= 0.02) & (x <= 5.12) & (y >= 0.02) & (y <= 5.12)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Calculate the point density based on the filtered data
    xy = np.vstack([x_filtered, y_filtered])
    z = gaussian_kde(xy)(xy)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Scatter plot with color based on point density
    sc = ax.scatter(x_filtered, y_filtered, c=z, cmap='jet', alpha=0.3, s=1)

    # Add the x=y line
    max_val = max(x_filtered.max(), y_filtered.max())
    ax.plot([0.025, max_val], [0.025, max_val], color='red', linestyle='--')

    # Set logarithmic scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set the ticks for logarithmic scales
    ax.set_xticks([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12])
    ax.set_yticks([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12])

    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # Set axis labels and title
    plt.xlabel(x_axis_title, fontsize=16, fontweight='bold')
    plt.ylabel(y_axis_title, fontsize=16, fontweight='bold')
    plt.title(plot_title, fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Place the metrics at a suitable location (metrics from the original data)
    textstr = '\n'.join((
        f'Bias: {bias:.6f}',
        f'MAE: {mae:.2f}',
        f'RMSE: {rmse:.2f}',
        f'Correlation: {corr:.2f}'))

    # Add text to the plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Show the plot
    plt.show()


def plot_density_with_metrics_rain(x, y, x_axis_title, y_axis_title, plot_title):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Calculate metrics using all the data (unfiltered)
    bias = np.mean(y - x)
    mae = mean_absolute_error(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    corr = np.corrcoef(x, y)[0, 1]

    # Filter out points where x or y are less than 0.2 for plotting
    mask = (x > 0.2) & (y > 0.2)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Calculate the point density based on the filtered data
    xy = np.vstack([x_filtered, y_filtered])
    z = gaussian_kde(xy)(xy)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Scatter plot with color based on point density
    sc = ax.scatter(x_filtered, y_filtered, c=z, cmap='twilight', alpha=0.3, s=1)

    # Add the x=y line
    max_val = max(x_filtered.max(), y_filtered.max())
    ax.plot([0.2, max_val], [0.2, max_val], color='red', linestyle='--')

    # Set logarithmic scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set the ticks for logarithmic scales
    ax.set_xticks([0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])
    ax.set_yticks([0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])

    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # Set axis labels and title
    plt.xlabel(x_axis_title, fontsize=16, fontweight='bold')
    plt.ylabel(y_axis_title, fontsize=16, fontweight='bold')
    plt.title(plot_title, fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Place the metrics (calculated from all data) at a suitable location
    textstr = '\n'.join((
        f'Bias: {bias:.6f}',
        f'MAE: {mae:.2f}',
        f'RMSE: {rmse:.2f}',
        f'Correlation: {corr:.2f}'))

    # Add text to the plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Show the plot
    plt.show()






