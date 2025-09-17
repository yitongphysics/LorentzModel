import pandas as pd
import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def computeTransition(nProbes, jumpFile, tranFile, channelFile, tetraFile):
    n_steps = int(1e5)
    file = open(jumpFile, 'r')
    channel = pd.read_csv(channelFile)
    tetraInfo = pd.read_csv(tetraFile)

    print("Reading jump file...")

    line1 = file.readline()
    line2 = file.readline()

    tetraId = [int(i) for i in line1.split(' ')[:-1]]
    tetraId.append(-1)
    numProbes = [int(i) for i in line2.split(' ')[:-1]]
    numProbes.append(nProbes - sum(numProbes))

    # read jumps info
    lines = file.readlines()
    t = []
    s1 = []
    s2 = []
    for line in lines:
        data = line.strip().split()
        tTmp = int(data[0])

        for i in range(len(data[1:])):
            s1Tmp, s2Tmp = data[1:][i].split(',')
            t.append(tTmp)
            s1.append(int(s1Tmp))
            s2.append(int(s2Tmp))

    print("Finished reading")

    jumps = pd.DataFrame({'t': t, 's1': s1, 's2': s2})
    jumps = jumps.sort_values(by=['t', 's1', 's2'])

    # create population dataset
    z = [(x, y) for x, y in zip(tetraId, numProbes)]
    z.sort()
    tetraId, numProbes = zip(*z)
    tetraId = list(tetraId)
    numProbes = np.array(numProbes)

    tetraId_to_idx = {tid: idx for idx, tid in enumerate(tetraId)}

    jumps_sorted = jumps.sort_values(by=['t', 's1', 's2'])
    t_arr = jumps_sorted['t'].to_numpy()
    s1_arr = jumps_sorted['s1'].to_numpy()
    s2_arr = jumps_sorted['s2'].to_numpy()

    s1_idx = np.array([tetraId_to_idx[s] for s in s1_arr])
    s2_idx = np.array([tetraId_to_idx[s] for s in s2_arr])

    n_tetra = len(tetraId)
    pop_matrix = np.zeros((n_steps + 1, n_tetra), dtype=int)
    pop_matrix[0] = numProbes

    # For each time step, apply all jumps
    for t in range(n_steps):
        if t%(n_steps/10) == 0:
            print(t)
        pop_matrix[t + 1] = pop_matrix[t]
        mask = (t_arr == t)
        # Count net changes for each tetraId at this time step
        if np.any(mask):
            # For all jumps at this t, subtract 1 from s1, add 1 to s2
            np.subtract.at(pop_matrix[t + 1], s1_idx[mask], 1)
            np.add.at(pop_matrix[t + 1], s2_idx[mask], 1)

    population = pd.DataFrame(pop_matrix, columns=tetraId)


    N = population.iloc[int(1e4):].mean()
    jumps = jumps[jumps['t'] > int(1e4)]


    # Create a 2D histogram of (s1, s2) pairs over all jumps
    s1_indices = np.array([tetraId_to_idx[s] for s in jumps['s1']])
    s2_indices = np.array([tetraId_to_idx[s] for s in jumps['s2']])
    Jij = np.zeros((len(tetraId), len(tetraId)))
    np.add.at(Jij, (s1_indices, s2_indices), 1)



    Tj = jumps['t'].max() - jumps['t'].min() + 1
    N = N * Tj


    with np.errstate(divide='ignore', invalid='ignore'):
        Tij = np.where(N.values[:, None] == 0, 0, Jij / N.values[:, None])
    np.fill_diagonal(Tij, 0)
    Tij[np.arange(len(tetraId)), np.arange(len(tetraId))] = -Tij.sum(axis=1)

    channel = channel.merge(tetraInfo[['id','volume']].rename(columns={'volume': 'v_i'}), left_on='i', right_on='id').drop(columns=['id'])
    channel = channel.merge(tetraInfo[['id','volume']].rename(columns={'volume': 'v_j'}), left_on='j', right_on='id').drop(columns=['id'])


    channel['Tij'] = float(0)
    channel['Tji'] = float(0)

    tetraId_to_idx = {tid: idx for idx, tid in enumerate(tetraId)}
    i_idx = channel['i'].map(tetraId_to_idx)
    j_idx = channel['j'].map(tetraId_to_idx)
    channel['Tij'] = Tij[i_idx, j_idx]
    channel['Tji'] = Tij[j_idx, i_idx]

    transitionRate1 = channel[['area', 'v_i', 'Tij']].rename(columns={'v_i': 'v', 'Tij': 'T'})
    transitionRate2 = channel[['area', 'v_j', 'Tji']].rename(columns={'v_j': 'v', 'Tji': 'T'})
    transitionRate = pd.concat([transitionRate1, transitionRate2])


    transitionRate.to_csv(tranFile, index=False)



if __name__ == "__main__":
    print("Computing transition rate...")
    if len(sys.argv) > 5:
        nProbes = int(sys.argv[1])
        jumpFile = sys.argv[2]
        tranFile = sys.argv[3]
        channelFile = sys.argv[4]
        tetraFile = sys.argv[5]
        computeTransition(nProbes, jumpFile, tranFile, channelFile, tetraFile)
    else:
        print("No argument provided")