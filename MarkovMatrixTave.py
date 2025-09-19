import numpy as np
import random
from itertools import product
import networkx as nx
import os
import sys
import copy
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

def markovMatrix(filePath, numBeads, ratio, numLayer, cutoff):
    numBeads = int(numBeads)
    ratio = float(ratio)
    numLayer = int(numLayer)
    cutoff = float(cutoff)

    comFile = filePath + 'sample/' + str(numBeads) + '_' + str(ratio) + '_COM.data'
    matFile = filePath + 'sample/' + str(numBeads) + '_' + str(ratio) + '_transitionMatrix.data'
    print(matFile)
    Lx, Ly, Lz = numLayer, numLayer, numLayer
    midBox = (numLayer-1) // 2
    origin   = (midBox,midBox,midBox)
    box_size = 100

    tetraCOM = []
    tetraV = []
    with open(comFile, 'r') as file:
        for line in file:
            x, y, z, v0 = list(map(float, line.split()))
            tetraCOM.append([x, y, z])
            tetraV.append(v0)
    tetraCOM = np.array(tetraCOM)
    n_states = len(tetraCOM)

    # ===============================
    # 2. Build a dictionary of 27 rate blocks
    # ===============================
    def make_blocks_27(n_states):
        # Create dictionary keys for every displacement (dx,dy,dz) in {-1,0,1}^3.
        blocks = {}
        for dx, dy, dz in product((-1, 0, 1), repeat=3):
            blocks[(dx, dy, dz)] = np.zeros((n_states, n_states))
        
        # --- Define intra-image transitions (dx,dy,dz) = (0,0,0):
        with open(matFile, 'r') as file:
            for line in file:
                i, j, mij, mji, dx, dy, dz = line.split()
                i, j, dx, dy, dz = map(int, [i, j, dx, dy, dz])
                mij, mji = map(float, [mij, mji])
                blocks[(dx, dy, dz)][i, j] = mij
                blocks[(-dx, -dy, -dz)][j, i] = mji

        # --- Set the diagonal of the (0,0,0) block so that the total outflow is zero.
        # For conservation, the overall generator must satisfy: for every column j,
        #   sum_{(dx,dy,dz)} [blocks[(dx,dy,dz)]][:,j] == 0.
        # We enforce this by computing, for each j, the total rate out (summing over all blocks)
        # and then setting [blocks[(0,0,0)]][j,j] = - (total rate out).
        for j in range(n_states):
            total_out = sum(blocks[d][:, j].sum() for d in blocks)
            blocks[(0,0,0)][j, j] = -total_out
        return blocks

    def make_blocks_1(n_states):
        rows = []
        cols = []
        vals = []
        with open(matFile, 'r') as file:
            for line in file:
                i, j, mij, mji, dx, dy, dz = line.split()
                i, j, dx, dy, dz = map(int, [i, j, dx, dy, dz])
                if dx == 0 and dy==0 and dz==0:
                    mij, mji = map(float, [mij, mji])

                    rows.append(i)
                    cols.append(j)
                    vals.append(mij)

                    rows.append(j)
                    cols.append(i)
                    vals.append(mji)

        from scipy.sparse import coo_matrix
        blocks = {}
        blocks[(0,0,0)] = coo_matrix(
            (vals, (rows, cols)),
            shape=(n_states, n_states)
        )

        G = blocks[(0,0,0)].tocsc()  # convert to CSC for fast column sums
        outflow = np.array(G.sum(axis=0)).ravel()
        G.setdiag(-outflow)
        blocks[(0,0,0)] = G.tocsr()
        return blocks

    # Create the 27 blocks.
    blocks27 = make_blocks_1(n_states)
    print('n states = ', n_states)

    # ===============================
    # 3. Build the transition matrix
    # ===============================
    def make_transition_matrix(Lx, Ly, Lz, blocks27, n_states):
        total_states = n_states*Lx*Ly*Lz
        transitionMatrix = np.zeros((total_states, total_states))

        for key, matrix in blocks27.items():
            dx, dy, dz = key[0], key[1], key[2]
            for i in range(n_states):
                for j in range(n_states):
                    if matrix[i,j] == 0: continue
                    if j == i: continue
                    for boxX in range(Lx):
                        for boxY in range(Ly):
                            for boxZ in range(Lz):
                                stateI = i + boxX*9*n_states + boxY*3*n_states + boxZ*n_states
                                stateJ = j + (boxX+dx)*9*n_states + (boxY+dy)*3*n_states + (boxZ+dz)*n_states
                                if stateJ < total_states and stateJ >= 0:
                                    transitionMatrix[stateI, stateJ] = matrix[i, j]
        for i in range(total_states):
            transitionMatrix[i, i] = -transitionMatrix[i, :].sum()

        return transitionMatrix
    transitionMatrix0 = make_transition_matrix(Lx, Ly, Lz, blocks27, n_states)
    print("Matrix constructed!")

    # ===============================
    # 4. Build the COM
    # ===============================
    def make_complete_COM(Lx, Ly, Lz, n_states, tetraCOM, box_size):
        completeCOM = []
        for boxX in range(Lx):
            for boxY in range(Ly):
                for boxZ in range(Lz):
                    com0 = copy.deepcopy(tetraCOM)
                    com0[:,0] += box_size * boxX
                    com0[:,1] += box_size * boxY
                    com0[:,2] += box_size * boxZ
                    completeCOM.extend(np.array(com0))
        
        return np.array(completeCOM)
    completeCOM = make_complete_COM(Lx, Ly, Lz, n_states, tetraCOM, box_size)
    tetraV = tetraV * Lx * Ly * Lz
    tetraV = np.array(tetraV)
    print("COM constructed!")

    # 2. Build a directed graph: add an edge i→j whenever P[i,j] > 0
    n = transitionMatrix0.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(transitionMatrix0 > 0)
    G.add_edges_from(zip(rows, cols))
    sccs = list(nx.strongly_connected_components(G))
    sccs = sorted(sccs, key=len, reverse = True)
    sccs = [list(comp) for comp in sccs]
    print("Connected component constructed!")

    # keep only comp0
    comp0 = sccs[0]
    print('num of nodes = ',len(comp0))
    keepID = comp0
    transitionMatrix = transitionMatrix0[keepID][:,keepID]
    completeCOM = completeCOM[keepID]
    tetraV = tetraV[keepID]

    for j in range(len(keepID)):
        transitionMatrix[j, j] -= transitionMatrix[j, :].sum()

    residual = transitionMatrix.sum(axis=1)
    transitionMatrix[np.diag_indices_from(transitionMatrix)] -= residual

    # ===============================
    # 6. compute e^M and diagonalize it
    # ===============================
    A = torch.as_tensor(transitionMatrix, dtype=torch.float64, device=device)
    eigenvalues, eigenvectors = torch.linalg.eig(A)

    P = eigenvectors          # Matrix of eigenvectors
    P_inv = torch.linalg.inv(P)

    # initialization
    t_plot = np.logspace(-10, 8, 100)
    cnt = 0
    msdFile = filePath + 'msd/' + str(numBeads) + '_' + str(ratio) + '.msd'
    print(msdFile)
    with open(msdFile, 'w') as f:
        print(' '.join(map(str, t_plot)), file=f, flush=True)

        minIDList = [i for i in range(len(comp0))]
        random.shuffle(minIDList)
        for i, minID in enumerate(minIDList):
            if np.max(np.abs(completeCOM[minID] - np.array([(1/2 + origin[0]) * box_size, (1/2 + origin[1]) * box_size, (1/2 + origin[2]) * box_size]))) > box_size * cutoff:
                continue
            cnt += 1
            print(i, cnt)
            
            p0 = torch.zeros(len(keepID), dtype=torch.float64, device=device)
            p0[minID] = 1.0

            dCOM = completeCOM - completeCOM[minID]
            dMSD = np.sum(dCOM ** 2, axis=1)

            MSD = np.zeros_like(t_plot)
            MSD = np.append([tetraV[minID]], MSD)
            for i, t in enumerate(t_plot):
                E_vec = torch.exp(eigenvalues * t)
                tmp = p0.to(dtype=P.dtype) @ P
                tmp *= E_vec 
                p = (tmp @ P_inv).real
                p[p < 0] = 0

                # compute MSD
                MSD[i+1] = np.dot(dMSD, p.cpu().numpy())

            print(' '.join(map(str, MSD)), file=f, flush=True)


if __name__ == "__main__":
    # Check if at least one argument is provided
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        numBeads = sys.argv[2]
        ratio = sys.argv[3]
        numLayer = sys.argv[4]
        cutoff = sys.argv[5]
        markovMatrix(filepath, numBeads, ratio, numLayer, cutoff)
    else:
        print("No argument provided")
