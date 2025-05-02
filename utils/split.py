import numpy as np
import pandas as pd

def split_train_val_test(df):
    # 1. Define year splits
    train_years = list(range(2004, 2016))  # 12 years
    val_years = list(range(2016, 2020))    # 4 years
    test_years = list(range(2020, 2024))    # 4 years
    
    train_data = df[df["Year"].isin(train_years)]
    val_data = df[df["Year"].isin(val_years)]
    test_data = df[df["Year"].isin(test_years)]

    return train_data, val_data, test_data

def random_sample_train_val_test(df):    
    scenario_vars = ['PlantingDay', 'Treatment', 'NFirstApp', 'IrrgDep', 'IrrgThresh']
    groups = df.groupby(scenario_vars, observed=True, sort=False)
    unique_groups = list(groups.groups.keys())
    np.random.shuffle(unique_groups)
    
    # Shuffle and split 70-15-15
    n = len(unique_groups)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    print(f"Scenarios: train({n_train * 12}), val({n_val * 4}), test({(n - n_train - n_val) * 4})")

    train_set = unique_groups[:n_train]
    val_set = unique_groups[n_train:n_train + n_val]
    test_set = unique_groups[n_train + n_val:]
    
    # Split data between train, val, test fro years to reduce matching
    train_data, val_data, test_data = split_train_val_test(df)
    
    def _filter_scenarios(_data, _dset):
        _dset = pd.DataFrame(_dset, columns=scenario_vars)
        for col in scenario_vars:
            if _data[col].dtype.name == 'category':
                _dset[col] = _dset[col].astype(_data[col].dtype)
        return _data.merge(_dset, on=scenario_vars, how='inner')
    
    return (_filter_scenarios(train_data, train_set), 
            _filter_scenarios(val_data, val_set), 
            _filter_scenarios(test_data, test_set))