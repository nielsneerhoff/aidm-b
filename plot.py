import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

DATA_ROOT = 'river-swim/'

NUM_RESULTS = 250

###################### MBIE & EB #####################
baseline_files = glob.glob(DATA_ROOT + '/*.dat')
mbie = {}
mbie_eb = {}
for file in baseline_files:

    # File is mbie-eb.
    if file.startswith('river-swim/mbie-eb'):
        metric = file.replace('river-swim/mbie-eb_', '').replace('.dat', '')
        if metric != 'sample_complexity':
            data = pd.read_csv(file, sep = r'\t+', engine = 'python')
            columns = np.array(data.columns)
            mbie_eb[metric] = {}
            mbie_eb[metric]['mean'] = np.array(data[columns[1]])
            mbie_eb[metric]['std'] = np.array(data[columns[2]]) - mbie_eb[metric]['mean']

    # File is mbie.
    else:
        metric = file.replace('river-swim/mbie_', '').replace('.dat', '')
        if metric != 'sample_complexity':
            data = pd.read_csv(file, sep = r'\t+', engine = 'python')
            columns = np.array(data.columns)
            mbie[metric] = {}
            mbie[metric]['mean'] = np.array(data[columns[1]])
            mbie[metric]['std'] = np.array(data[columns[2]]) - mbie[metric]['mean']

######################################################

###################### Max- opt ######################
mediator_max_opt_files = glob.glob(DATA_ROOT + 'mediator-max-opt/*.dat')
mediator_max_opt = {}

for file in mediator_max_opt_files:
    filename = file.replace(
        'river-swim/mediator-max-opt/mediator-max-opt-', '')
    info = filename.split('-')
    offset = float(info[0])
    rho = float(info[1].split('_')[0])
    metric = info[1].replace(f'{str(rho)}_', '').replace('.dat', '')
    if metric not in (
        'sample_complexity','coverage_error_squared_R', 'reward_timeline'):
        data = pd.read_csv(file, sep = r'\t+', engine = 'python')
        columns = np.array(data.columns)

        # Traverse to find right dict.
        if metric not in mediator_max_opt:
            mediator_max_opt[metric] = {}
        if offset not in mediator_max_opt[metric]:
            mediator_max_opt[metric][offset] = {}
        if rho not in mediator_max_opt[metric][offset]:
            mediator_max_opt[metric][offset][rho] = {}

        mediator_max_opt[metric][offset][rho]['mean'] = \
            np.array(data[columns[1]])
        mediator_max_opt[metric][offset][rho]['std'] \
            = np.array(data[columns[2]]) - np.array(data[columns[1]])

######################################################

####################### Random #######################
mediator_random_files = glob.glob(DATA_ROOT + 'mediator-random/*.dat')
mediator_random = {}

for file in mediator_random_files:
    filename = file.replace(
        'river-swim/mediator-random/mediator-random-', '')
    info = filename.split('-')
    offset = float(info[0])
    rho = float(info[1].split('_')[0])
    metric = info[1].replace(f'{str(rho)}_', '').replace('.dat', '')
    if metric not in (
        'sample_complexity','coverage_error_squared_R', 'reward_timeline', 'state_action_count'):
        data = pd.read_csv(file, sep = r'\t+', engine = 'python')
        columns = np.array(data.columns)

        # Traverse to find right dict.
        if metric not in mediator_random:
            mediator_random[metric] = {}
        if offset not in mediator_random[metric]:
            mediator_random[metric][offset] = {}
        if rho not in mediator_random[metric][offset]:
            mediator_random[metric][offset][rho] = {}

        mediator_random[metric][offset][rho]['mean'] = \
            np.array(data[columns[1]])
        mediator_random[metric][offset][rho]['std'] \
            = np.array(data[columns[2]]) - np.array(data[columns[1]])

######################################################

# metric = 'cumulative_instantaneous_loss' # CHANGE FOR PLOT.
metric = 'cumulative_rewards' # CHANGE FOR PLOT.

offsets = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
rhos = [0.02, 0.04, 0.08, 0.16, 0.32]

NUM_ROWS = 6 # offset
NUM_COLS = 5 # rho

fig, axs = plt.subplots(
    NUM_ROWS, NUM_COLS, sharex=True, sharey=True)
for r in range(NUM_ROWS):
    for c in range(NUM_COLS):
        rho = rhos[c]
        offset = offsets[r]
        max_opt = mediator_max_opt[metric][offset][rho]['mean'][:NUM_RESULTS]
        random = mediator_random[metric][offset][rho]['mean'][:NUM_RESULTS]
        mbie_data = mbie[metric]['mean'][:NUM_RESULTS]
        mbie_eb_data = mbie_eb[metric]['mean'][:NUM_RESULTS]
        index = range(NUM_RESULTS)
        sns.lineplot(x = index, y = max_opt, ax = axs[r, c], color = 'red', alpha = 0.5)
        sns.lineplot(x = index, y = random, ax = axs[r, c], color = 'orange', alpha = 0.5)
        sns.lineplot(x = index, y = mbie_data, ax = axs[r, c], color = 'blue', alpha = 0.5)
        sns.lineplot(x = index, y = mbie_eb_data, ax = axs[r, c], color = 'cyan', alpha = 0.5)
        axs[r, c].set_title(f'({offset}, {rho})')
        # axs[r, c].set_yscale('log')
plt.show()