"""TODO: Add module docstring describing this check script."""

import os
import pandas as pd
import glob

dirpath = os.path.join('datasets_final','chanlam_final','progressive_by_reactants')
files = sorted(glob.glob(os.path.join(dirpath,'train_rows_*.csv')),
               key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]))

if not files:
    print('No subset files found in', dirpath)
    raise SystemExit(1)

sets = []
rows = []
for p in files:
    df = pd.read_csv(p)
    # use full row tuples as identity
    tup_set = set(tuple(x) for x in df.fillna('').values)
    sets.append((p, tup_set))
    rows.append((p, len(df)))

print('Found files and row counts:')
for p, r in rows:
    print('-', p, r)

# check nestedness: for each i<j, check if set_i subset of set_j
all_nested = True
for i in range(len(sets)):
    pi, si = sets[i]
    for j in range(i+1, len(sets)):
        pj, sj = sets[j]
        if si.issubset(sj):
            if si == sj:
                print(f'{os.path.basename(pi)} IS EQUAL to {os.path.basename(pj)}')
            else:
                print(f'{os.path.basename(pi)} is subset of {os.path.basename(pj)}')
        else:
            all_nested = False
            missing = si - sj
            print(f'{os.path.basename(pi)} is NOT subset of {os.path.basename(pj)}; {len(missing)} rows missing')
            # print up to 3 examples
            for k, ex in enumerate(list(missing)[:3]):
                print('  example missing row:', ex)

print('\nOverall nestedness:', 'YES' if all_nested else 'NO')
