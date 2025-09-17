import pandas as pd
import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def MarkovArray(nFrames, areaFile, filePath, outputFile):
    pExpected = []
    pRealized = []
    velocityCorrelation = []

    channel = pd.read_csv(areaFile)
    channel['area_sum'] = channel.groupby('i')['area'].transform('sum')
    channel['P'] = channel['area'] / channel['area_sum']
    

    for nFrame in range(1, nFrames):
        file = filePath + "_" + str(nFrame) + ".pos"
        
        try:
            jumps = pd.read_csv(file, sep=' ', header=None)
        except:
            velocityCorrelation.append(np.nan)
            pExpected.append(np.nan)
            pRealized.append(np.nan)
            continue


        jumps.columns = ['ci', 't_curr', 't_next', 'px', 'py', 'pz']

        jumps['t_prev'] = jumps.groupby('ci')['t_curr'].shift(1)
        jumps['px_prev'] = jumps.groupby('ci')['px'].shift(1)
        jumps['py_prev'] = jumps.groupby('ci')['py'].shift(1)
        jumps['pz_prev'] = jumps.groupby('ci')['pz'].shift(1)

        jumps = jumps[['t_prev', 't_curr', 't_next', 'px', 'py', 'pz', 'px_prev', 'py_prev', 'pz_prev']]
        jumps.dropna(inplace=True)
        jumps = jumps.astype({'t_prev': 'int32', 't_curr': 'int32', 't_next': 'int32'})

        jumps['corr'] = (jumps['px']*jumps['px_prev'] + jumps['py']*jumps['py_prev'] + jumps['pz']*jumps['pz_prev'])
        velocityCorrelation.append(jumps['corr'].mean())

        jumps = jumps.merge(channel[['i', 'j', 'P']], left_on=['t_curr', 't_prev'], right_on=['i','j'], how='left')
        jumps = jumps[['t_prev', 't_curr', 't_next', 'P']]
        jumps.dropna(inplace=True)

        pExpected.append(jumps['P'].mean())
        pRealized.append(len(jumps[jumps['t_next'] == jumps['t_prev']])/len(jumps))


    result = pd.DataFrame({'velocityCorrelation': velocityCorrelation, 'pExpected': pExpected, 'pRealized': pRealized})
    result.to_csv(outputFile, index=False)


if __name__ == "__main__":
    print("Computing transition rate...")
    if len(sys.argv) > 3:
        nFrames = int(sys.argv[1])
        areaFile = sys.argv[2]
        filePath = sys.argv[3]
        outputFile = sys.argv[4]
        MarkovArray(nFrames, areaFile, filePath, outputFile)
    else:
        print("No argument provided")