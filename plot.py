import glob
import pandas as pd

DATA_ROOT = 'river-swim/'

###################### MBIE & EB #####################
baseline_files = glob.glob(DATA_ROOT + '/*')
mbie = {}
mbie_eb = {}
for file in baseline_files:

    # File is mbie-eb.
    if file.startswith('mbie_eb'):
        metric = file.replace('mbie_eb_', '\t')
        data = pd.read_fwf(file)
        mbie_eb[metric] = data

    # File is mbie.
    else:
        metric = file.replace('river-swim/mbie_', '').replace('.dat', '')
        data = pd.read_csv(file, sep = r'\t+')
        print(data.iloc[:, 4])
        mbie_eb[metric] = {}
        mbie_eb[metric]['mean'] = data[0]
        mbie_eb[metric]['std'] = data['std']

a = 1 / 0

######################################################

###################### Max- opt ######################
mediator_max_opt_files = glob.glob(DATA_ROOT + '/mediator-max-opt')
mediator_max_opt = {}

for file in mediator_max_opt_files:
    filename = file.replace('mediator-max-opt-', '')
    info = filename.split('-')
    data = pd.read_csv(file)
    # Info 0 contains offset, info 1 contains rho, info 2 contains metric.
    mediator_max_opt[info[0]][info[1]][info[2]] = data

mediator_random_dir = DATA_ROOT + '/mediator-random'