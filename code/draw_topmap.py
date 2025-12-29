"""
Topographical Map of the Brain Plotting (Relative Power), Fig4,5
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import matplotlib
import os
from spectral_connectivity import Multitaper, Connectivity
from collections import defaultdict
import json
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use("Qt5Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
print(np.__version__)

plt.ion()  # if not showing figure: plt.ioff()
# adjust figure format
plt.rc('font', size=9, family='Arial', weight='normal')
matplotlib.rcParams['axes.labelsize'] = 8
matplotlib.rcParams['axes.labelweight'] = 'normal'
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['axes.titlesize'] = 9
matplotlib.rcParams['axes.titleweight'] = 'normal'
matplotlib.rcParams['axes.linewidth'] = 1.0
matplotlib.rcParams['svg.fonttype'] = 'none'

def load_and_clean_csv(csv_file):
    # Read CSV with default headers
    df = pd.read_csv(csv_file)

    # Generate new column names
    new_columns = [
        f'Column_{i}' if 'Unnamed' in str(col) else col
        for i, col in enumerate(df.columns)
    ]

    # The last two columns are none, which need to add back
    i = len(new_columns) + 1
    new_columns.append(f'Column_{i}')
    i = i + 1
    new_columns.append(f'Column_{i}')

    # Reload with cleaned headers
    df_clean = pd.read_csv(csv_file, header=0, names=new_columns)

    return df_clean

def find_indices_for_mark(raw, sfreq, target_name):
    """
    Find event indices for a specific marker value from struct_event

    Parameters:
    struct_event : List of event dictionaries (from get_event_info)
    target_mark : int, marker value to search for (default=1)

    Returns:
    numpy.ndarray: array of [start, end] indices for the target marker
                   Empty array if marker not found
    """
    # Read all events contained in the EEG signals
    annotations = raw.annotations

    # Find the indices of all target events
    target_indices = [i for i, desc in enumerate(annotations.description)
                      if desc == target_name]

    # The sampling points at which the target events occur
    eve_index = []
    for i in range(len(target_indices)):
        start_i = annotations.onset[target_indices[i]] * sfreq
        if target_indices[i] == (annotations.onset.shape[0] - 1):   # 已经是最后一个事件的情况
            end_i = raw.get_data().shape[1] - 1
        else:
            end_i = annotations.onset[target_indices[i] + 1]  * sfreq - 1
        eve_index.append([start_i, end_i])

    eve_index = np.array(eve_index)

    return eve_index

def topmap_all(raw, sfreq):
    """
    Calculate the frequency band relatve power
    """
    all_features = defaultdict(lambda: defaultdict(dict))

    # frequency band
    frequency_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }

    target_event = ['Resting task', 'Counting task 1', 'Word formation task', 'Counting task 2', 'Imaginary defecation task',
                    'Simulated defecation task', 'Imaginary anal contraction task', 'Simulated anal contraction task']
    for event_name in target_event:
        print(event_name)
        eve_indices = find_indices_for_mark(raw, sfreq, event_name)

        # For the case of multiple trials, calculate the mean value of the trial features
        trail_num = eve_indices.shape[0]

        for band, frequency in frequency_bands.items():
            if trail_num > 1:
                power_trail_all = []
                for t_num in range(trail_num):

                    eve_indices_num = eve_indices[t_num]
                    eve_raw = raw.copy().crop(eve_indices_num[0] / sfreq, eve_indices_num[1] / sfreq)
                    psd = eve_raw.compute_psd(
                        method="welch",
                        fmin=0.5,
                        fmax=45,
                    )

                    psd_data = psd.get_data()
                    freq_mask = np.logical_and(psd.freqs >= frequency[0], psd.freqs <= frequency[1])
                    band_power = np.sum(psd_data[:, freq_mask], axis = 1)
                    total_power = np.sum(psd_data[:, :], axis=1)
                    power_trail_all.append(band_power / total_power)
                power_trail_all = np.array(power_trail_all)
                power_mean = np.mean(power_trail_all, axis=0)

            else:
                eve_indices_num = eve_indices[0]
                # Only the first 2 minutes of resting-state data were extracted
                if event_name == 'Resting task':
                    eve_raw = raw.copy().crop(eve_indices_num[0] / sfreq,
                                              min(eve_indices_num[0] / sfreq + 120, eve_indices_num[1] / sfreq))
                else:
                    eve_raw = raw.copy().crop(eve_indices_num[0] / sfreq, eve_indices_num[1] / sfreq)

                psd = eve_raw.compute_psd(
                    method="welch",
                    fmin=0.5,
                    fmax=45,
                )
                # Calculate the power of each frequency band
                psd_data = psd.get_data()
                freq_mask = np.logical_and(psd.freqs >= frequency[0], psd.freqs <= frequency[1])
                band_power = np.sum(psd_data[:, freq_mask], axis=1)
                total_power = np.sum(psd_data[:, :], axis=1)
                power_mean = band_power / total_power

            all_features[event_name][band] = list(power_mean)

    return all_features

def draw_topmap(task_name, power_db, info, band, type, ax, fig):
    """
    plot topmsp
    """
    frequency_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }
    [low_freq, high_freq] = frequency_bands[band]

    # Adjust the electrode positions
    info1 = info.copy()
    for ch in info1['chs']:
        ch['loc'][1] *= 1.1
        ch['loc'][1] -= 0.025

    # plot topmsp
    im, _ = mne.viz.plot_topomap(power_db, info1, sphere=(0, 0, 0, 0.1), contours=1, outlines='head',axes=ax, cmap=create_custom_colormap())

    # Adjust the colorbar range
    if band == 'Alpha':
        clim = {
            'Resting task': (0, 0.3),
            'Counting task 1': (0, 0.15),
            'Word formation task': (0, 0.15),
            'Counting task 2': (0, 0.15),
            'Imaginary defecation task': (0, 0.22),
            'Simulated defecation task': (0, 0.22),
            'Imaginary anal contraction task': (0, 0.25),
            'Simulated anal contraction task': (0, 0.22),
        }
        clim_task = clim[task_name]
        im.set_clim(clim_task[0], clim_task[1])
    else:
        clim = {
            'Delta': (0, 1),
            'Theta': (0, 0.3),
            'Alpha': (0, 0.25),
            'Beta': (0, 0.2),
            'Gamma': (0, 0.05)
        }
        clim_band = clim[band]
        im.set_clim(clim_band[0], clim_band[1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    clb = fig.colorbar(ax.images[-1], cax=cax, shrink=0.61)

    clb.ax.tick_params(labelsize=8)
    if type == 'Controls':
        cax.set_visible(False)

    ax.set_xlabel(type, fontsize=8.5)

    return fig

def create_custom_colormap():
    """Create a custom smooth blue-white-red colormap"""
    center_pos = 1 / 2
    cdict = {
        'red': [(0.0, 0.035, 0.035),
                (center_pos - 0.2, 0.4, 0.4),
                (center_pos, 1.0, 1.0),
                (center_pos + 0.2, 1.0, 1.0),
                (1.0, 0.8, 0.8)],

        'green': [(0.0, 0.18, 0.18),
                  (center_pos - 0.2, 0.6, 0.6),
                  (center_pos, 1.0, 1.0),
                  (center_pos + 0.2, 0.3, 0.3),
                  (1.0, 0.0, 0.0)],

        'blue': [(0.0, 0.4, 0.4),
                 (center_pos - 0.2, 0.8, 0.8),
                 (center_pos, 1.0, 1.0),
                 (center_pos + 0.2, 0.2, 0.2),
                 (1.0, 0.0, 0.0)]
    }

    smooth_blue_red = mcolors.LinearSegmentedColormap('smooth_blue_red', cdict)

    return smooth_blue_red


if __name__ == "__main__":
    '''
    ##********************************* Calculate the relative power of each channel ********************************************##
    type = 'Patients'   # 'Controls'
    dir_path = ''  # EEG data path
    set_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.set')]  # The EEG data used in this study are in .set format.

    for file in set_files:
        sub_name = file.split('.')[0]
        print(file)
        raw = mne.io.read_raw_eeglab(dir_path + file, preload=True)
        raw.plot(duration=30)
        sfreq = raw.info['sfreq']

        # Calculate and save the relative power of each channel
        all_features = topmap_all(raw, sfreq)
        filename = '../data/relative_power_topmap/' + type + '/' + sub_name + '.json'
        with open(filename, 'w') as f:
            json.dump(all_features, f, indent=4)
    '''

    ##************************************ Plot the topographic map of relative power ****************************************##
    # Channel information
    sfreq = 250
    eeg_ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'AF4', 'FC3', 'FC4', 'CP3', 'POz', 'FCz', 'Fpz', 'PO8', 'CP2', 'CP1', 'FC2', 'FC1', 'C3', 'C4', 'P3', 'P4', 'CP4', 'PO3', 'PO4', 'F5', 'TP9', 'FT10', 'FT9', 'CPz', 'CP6', 'CP5', 'FC6', 'FC5', 'O1', 'O2', 'F7', 'F8', 'F6', 'C5', 'C6', 'P5', 'AF3', 'P2', 'P1', 'C2', 'C1', 'F2', 'F1', 'TP10', 'T7', 'T8', 'P7', 'P8', 'P6', 'AF7', 'AF8', 'FT7', 'TP8', 'PO7', 'TP7', 'FT8', 'Oz', 'Pz', 'Cz', 'Fz']
    n_channels = 64
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=list(eeg_ch_names), sfreq=sfreq, ch_types=ch_types)
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # dictionary for storing feature values
    all_features = defaultdict(
        lambda: defaultdict(
                lambda: defaultdict(list)
        )
    )

    # Read the previously extracted features
    type = ['Patients', 'Controls']
    for type_i in type:
        dir_path = '../data/relative_power_topmap/' + type_i + '/'
        folder = Path(dir_path)
        all_data = {f.stem: json.load(open(f)) for f in folder.glob("*.json")}

        for key_all, value_all in all_data.items():
            for task_name, value_task in value_all.items():
                for band_name, band_value in value_task.items():
                    all_features[type_i][task_name][band_name].append(band_value)

    # Plot the topographic map of relative power
    task_names = list(all_features['Patients'].keys())
    for band in list(all_features['Patients']['Resting task'].keys()):
        fig = plt.figure(figsize=(16/2.54, 20/2.54))
        gs = gridspec.GridSpec(
            nrows=4,
            ncols=5,
            width_ratios=[1, 1, 0.3, 1, 1],
            wspace=0.1,
            hspace=0.1
        )

        axes = []
        for row in range(4):
            for col in range(5):
                ax = fig.add_subplot(gs[row, col])
                axes.append(ax)

        i = 0
        char_num = 0
        for task_name, value_task in all_features['Patients'].items():
            # Control
            band_value = all_features['Controls'][task_name][band]
            band_name = band
            band_value1 = np.mean(np.array(band_value), axis=0)
            fig = draw_topmap(task_name, band_value1, info, band_name, type='Controls', ax=axes[i], fig=fig)

            # Patients
            band_value = value_task[band]
            band_name = band
            band_value1 = np.mean(np.array(band_value), axis=0)
            fig = draw_topmap(task_name, band_value1, info, band_name, type='Patients', ax=axes[i + 1], fig=fig)

            # Title: task
            title1 = fig.text(
                x=(axes[i].get_position().x1 + axes[i+1].get_position().x0) / 2 - 0.01,
                y=axes[i].get_position().y1 - 0.028,
                s=task_name,
                ha='center',
                va='bottom',
                fontsize=9
            )

            # Subplot index
            label = f'({chr(97 + char_num)})'
            title2 = fig.text(
                x=(axes[i].get_position().x1 + axes[i + 1].get_position().x0) / 2 - 0.01,
                y=axes[i].get_position().y0 -0.005,
                s=label,
                ha='center',
                va='bottom',
                fontsize=9
            )
            char_num += 1
            i += 2

            # Skip when encountering the third column (leave a blank space)
            if (i-2) % 5 == 0:
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                axes[i].set_xlabel('')
                axes[i].set_ylabel('')

                for spine in axes[i].spines.values():
                    spine.set_visible(False)
                i += 1

        plt.show()
        print('a')
    print('a')

