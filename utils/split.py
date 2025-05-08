import numpy as np
import pandas as pd

def split_train_val_test(
        df, train_years=[2004,2016],
        val_years=[2016, 2020],
        test_years=[2020, 2024]
    ):
    # 1. Define year splits
    tr_yrs = list(range(*train_years))
    vl_yrs = list(range(*val_years))
    ts_yrs = list(range(*test_years))
    
    train_data = df[df["Year"].isin(tr_yrs)]
    val_data = df[df["Year"].isin(vl_yrs)]
    test_data = df[df["Year"].isin(ts_yrs)]

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