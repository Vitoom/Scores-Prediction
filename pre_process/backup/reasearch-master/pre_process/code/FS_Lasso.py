
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

'''
# input the certain csv files as DataFrame
# columns: SnapId , ...load score, load level, performance score, performance level
# float, ... , int,...   float       object          float             object
# (the length 46)
# index: 0, 1, 2, ... (the length 1510)
# then convert the DataFrame to numpy.array
'''
df_db = pd.read_csv('../csv/DBID(172908691)_INSTID(1).csv')
arr_db = df_db.values

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
    
sample_db = arr_db[:, 1:-5]
target_db = arr_db[:, -4:-1]
# choose the performance score as target
target = target_db[:, 2]

for i in range(sample_db.shape[0]):
    for j in range(sample_db.shape[1]):
        # process the NaN
        if sample_db[i][j] == 'null':
            # assign 1.2 or other value
            sample_db[i][j] = 1.2
pd.set_option('display.max_colwidth', -1)
#pd.set_option('expand_frame_repr', True)
print(sample_db)
'''
StandardScaler for normalization
'''
'''
scaler1 = StandardScaler()
scaler1.fit(sample_db)
sample_db = scaler1.transform(sample_db)

scaler2 = StandardScaler()
scaler2.fit(target)
target = scaler2.transform(target)
'''



'''
by LassoCV 
CV: Cross-Validation for determining the coefficient of the L1-norm
'''

lassocv = LassoCV(max_iter=2000, normalize=True)
lassocv.fit(sample_db, target)

# the coefficient of the L1-norm
print("lamda = ", lassocv.alpha_)

# weight matrix
print("the weight matrix: ")
print(lassocv.coef_,lassocv.coef_.shape)

print("number of the chosen features: ", np.sum(lassocv.coef_ != 0))

# the features been chosen
mask = lassocv.coef_ != 0
print("the chosen features:")
print(df_db.columns[1:-5][mask])

























