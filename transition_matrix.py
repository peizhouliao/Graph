import numpy as np
import pandas as pd

df = pd.read_table('state_corr.dat', sep=',', names=['dbid', 'region', 'state'])

## Divisions with states
new_england = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT']
mid_atlantic = ['NJ', 'NY', 'PA']
east_north_central = ['IN', 'IL', 'MI', 'OH', 'WI']
west_north_central = ['IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD']
south_atlantic = ['DE', 'DC', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV']
east_south_central = ['AL', 'KY', 'MS', 'TN']
west_south_central = ['AR', 'LA', 'OK', 'TX']
mountain = ['AZ', 'CO', 'ID', 'NM', 'MT', 'UT', 'NV', 'WY']
pacific = ['CA', 'OR', 'WA','AK', 'HI']

states = new_england + mid_atlantic + east_north_central + west_north_central + south_atlantic + \
    east_south_central + west_south_central + mountain + pacific

## obtain the claimed state and true state
df['status'] = df['region'].apply(lambda x: (str(x).startswith('region') and x[-2:]) in states)
df = df[df.status==1]
df['code'] = df['region'].apply(lambda x: x[-2:])
df['code1'] = df['state'].apply(lambda x: x.strip('()'))

## contingency table for states
data = df.groupby(['code', 'code1']).size().reset_index(name='counts')
code_sum = data.groupby('code')['counts'].sum().to_dict()

dic = {state:i for i,state in enumerate(sorted(code_sum.keys()))}

## make the transition matrix
n_state = len(dic)
matrix_p = np.zeros(shape=(n_state, n_state))
for i, row in data.iterrows():
    if row['code1'] in dic and row['code'] in dic and row['code'] in code_sum:
        matrix_p[dic[row['code1']]][dic[row['code']]] = row['counts']/float(code_sum[row['code']])

matrix_p.dump('transition_matrix.dat')