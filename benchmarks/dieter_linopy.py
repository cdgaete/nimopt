#!/usr/bin/env python3
"""
DIETER Energy System Model - linopy Implementation

For benchmarking comparison with nimopt.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import linopy

DATA_DIR = Path(__file__).parent.parent / "examples" / "dieter" / "data"
CURTAILMENT_COST = 0.1
INFEAS_COST = 10000


def load_data(n_hours: int = 8760):
    """Load all data from CSV files."""
    tech_df = pd.read_csv(DATA_DIR / "technologies.csv")
    tech_params = pd.read_csv(DATA_DIR / "tech_params.csv").set_index('Technologies')
    sto_params = pd.read_csv(DATA_DIR / "sto_params.csv").set_index('Storages')
    load_df = pd.read_csv(DATA_DIR / "load.csv")
    avail_df = pd.read_csv(DATA_DIR / "availability.csv")

    # Sets
    dispatchable = tech_df[tech_df['Dispatchable'] == 1]['Technologies'].tolist()
    non_dispatchable = tech_df[tech_df['Dispatchable'] == 0]['Technologies'].tolist()
    storages = sto_params.index.tolist()
    hours = list(range(1, n_hours + 1))

    # Filter availability to relevant techs and hours
    avail_df = avail_df[
        (avail_df['Technologies'].isin(non_dispatchable)) &
        (avail_df['Hours'] <= n_hours)
    ]

    # Build availability as xarray
    avail_matrix = np.zeros((len(non_dispatchable), n_hours))
    for i, t in enumerate(non_dispatchable):
        tech_data = avail_df[avail_df['Technologies'] == t].sort_values('Hours')
        avail_matrix[i, :] = tech_data['Availability'].values
    availability = xr.DataArray(
        avail_matrix,
        dims=['NonDispatchable', 'Hours'],
        coords={'NonDispatchable': non_dispatchable, 'Hours': hours}
    )

    return {
        'dispatchable': dispatchable,
        'non_dispatchable': non_dispatchable,
        'storages': storages,
        'hours': hours,
        'tech_params': tech_params,
        'sto_params': sto_params,
        'load': xr.DataArray(load_df.loc[:n_hours-1, 'Load'].values, 
                             dims=['Hours'], coords={'Hours': hours}),
        'availability': availability,
    }


def build_model(data: dict, corr_factor: float = 1.0):
    """Build the DIETER linopy model."""
    m = linopy.Model()

    disp = data['dispatchable']
    nd = data['non_dispatchable']
    sto = data['storages']
    hours = data['hours']

    tech_params = data['tech_params']
    sto_params = data['sto_params']
    load = data['load']
    avail = data['availability']

    # Parameters as xarray DataArrays
    inv_cost_disp = xr.DataArray([tech_params.loc[t, 'InvestmentCost'] for t in disp],
                                  dims=['Dispatchable'], coords={'Dispatchable': disp})
    inv_cost_nd = xr.DataArray([tech_params.loc[t, 'InvestmentCost'] for t in nd],
                                dims=['NonDispatchable'], coords={'NonDispatchable': nd})
    marg_cost_disp = xr.DataArray([tech_params.loc[t, 'MarginalCost'] for t in disp],
                                   dims=['Dispatchable'], coords={'Dispatchable': disp})
    fixed_cost_disp = xr.DataArray([tech_params.loc[t, 'FixedCost'] for t in disp],
                                    dims=['Dispatchable'], coords={'Dispatchable': disp})
    fixed_cost_nd = xr.DataArray([tech_params.loc[t, 'FixedCost'] for t in nd],
                                  dims=['NonDispatchable'], coords={'NonDispatchable': nd})
    max_install_nd = xr.DataArray([tech_params.loc[t, 'MaxInstallable'] for t in nd],
                                   dims=['NonDispatchable'], coords={'NonDispatchable': nd})

    inv_cost_energy = xr.DataArray([sto_params.loc[s, 'InvestmentCostEnergy'] for s in sto],
                                    dims=['Storages'], coords={'Storages': sto})
    inv_cost_power = xr.DataArray([sto_params.loc[s, 'InvestmentCostPower'] for s in sto],
                                   dims=['Storages'], coords={'Storages': sto})
    marg_cost_sto = xr.DataArray([sto_params.loc[s, 'MarginalCost'] for s in sto],
                                  dims=['Storages'], coords={'Storages': sto})
    fixed_cost_sto = xr.DataArray([sto_params.loc[s, 'FixedCost'] for s in sto],
                                   dims=['Storages'], coords={'Storages': sto})
    efficiency_sto = xr.DataArray([sto_params.loc[s, 'Efficiency'] for s in sto],
                                   dims=['Storages'], coords={'Storages': sto})
    max_energy_sto = xr.DataArray([sto_params.loc[s, 'MaxEnergy'] for s in sto],
                                   dims=['Storages'], coords={'Storages': sto})
    max_power_sto = xr.DataArray([sto_params.loc[s, 'MaxPower'] for s in sto],
                                  dims=['Storages'], coords={'Storages': sto})
    sqrt_eff = np.sqrt(efficiency_sto)

    # Variables
    G_D = m.add_variables(lower=0, dims=['Dispatchable', 'Hours'],
                          coords=[disp, hours], name='G_D')
    G_ND = m.add_variables(lower=0, dims=['NonDispatchable', 'Hours'],
                           coords=[nd, hours], name='G_ND')
    CU = m.add_variables(lower=0, dims=['NonDispatchable', 'Hours'],
                         coords=[nd, hours], name='CU')
    STO_IN = m.add_variables(lower=0, dims=['Storages', 'Hours'],
                             coords=[sto, hours], name='STO_IN')
    STO_OUT = m.add_variables(lower=0, dims=['Storages', 'Hours'],
                              coords=[sto, hours], name='STO_OUT')
    STO_L = m.add_variables(lower=0, dims=['Storages', 'Hours'],
                            coords=[sto, hours], name='STO_L')

    N_D = m.add_variables(lower=0, dims=['Dispatchable'], coords=[disp], name='N_D')
    N_ND = m.add_variables(lower=0, dims=['NonDispatchable'], coords=[nd], name='N_ND')
    N_STO_E = m.add_variables(lower=0, dims=['Storages'], coords=[sto], name='N_STO_E')
    N_STO_P = m.add_variables(lower=0, dims=['Storages'], coords=[sto], name='N_STO_P')
    G_INF = m.add_variables(lower=0, dims=['Hours'], coords=[hours], name='G_INF')

    # Objective
    obj = (
        (marg_cost_disp * G_D).sum() +
        (marg_cost_sto * (STO_IN + STO_OUT)).sum() +
        CURTAILMENT_COST * CU.sum() +
        INFEAS_COST * G_INF.sum() +
        corr_factor * (inv_cost_disp * N_D).sum() +
        corr_factor * (inv_cost_nd * N_ND).sum() +
        corr_factor * (inv_cost_energy * N_STO_E).sum() +
        corr_factor * (inv_cost_power * N_STO_P).sum() +
        corr_factor * (fixed_cost_disp * N_D).sum() +
        corr_factor * (fixed_cost_nd * N_ND).sum() +
        corr_factor * (fixed_cost_sto * 0.5 * (N_STO_P + N_STO_E)).sum()
    )
    m.add_objective(obj, sense='min')

    # Energy Balance
    m.add_constraints(
        G_D.sum('Dispatchable') + G_ND.sum('NonDispatchable') +
        STO_OUT.sum('Storages') + G_INF == STO_IN.sum('Storages') + load,
        name='EnergyBalance'
    )

    # Generation limits - dispatchable
    m.add_constraints(G_D <= N_D, name='MaxGenDisp')

    # Generation limits - non-dispatchable
    m.add_constraints(G_ND + CU == avail * N_ND, name='MaxGenNonDisp')

    # Max installable
    m.add_constraints(N_ND <= max_install_nd, name='MaxInstallND')

    # Storage constraints
    m.add_constraints(STO_IN <= N_STO_P, name='MaxWithdrawSto')
    m.add_constraints(STO_OUT <= N_STO_P, name='MaxGenSto')
    m.add_constraints(STO_L <= N_STO_E, name='MaxLevelSto')
    m.add_constraints(N_STO_E <= max_energy_sto, name='MaxEnergySto')
    m.add_constraints(N_STO_P <= max_power_sto, name='MaxPowerSto')

    # Storage balance with cyclic lag
    STO_L_prev = STO_L.roll(Hours=1)
    m.add_constraints(
        STO_L == STO_L_prev + sqrt_eff * STO_IN - STO_OUT / sqrt_eff,
        name='StorageBalance'
    )

    return m


def main():
    parser = argparse.ArgumentParser(description='DIETER - linopy')
    parser.add_argument('--hours', type=int, default=168)
    args = parser.parse_args()

    n_hours = args.hours
    corr_factor = n_hours / 8760

    print(f"\n{'='*60}")
    print(f"DIETER Energy System Model - linopy Implementation")
    print(f"{'='*60}")
    print(f"Hours: {n_hours} ({n_hours/24:.1f} days)")

    # Load data
    print("\nLoading data...")
    t0 = time.time()
    data = load_data(n_hours)
    t_load = time.time() - t0
    print(f"  Data loading: {t_load:.2f}s")

    # Build model
    print("\nBuilding model...")
    t0 = time.time()
    model = build_model(data, corr_factor)
    t_build = time.time() - t0
    print(f"  Model building: {t_build:.2f}s")

    # Solve with HiGHS
    print("\nSolving...")
    t0 = time.time()
    model.solve(solver_name='highs', log_fn=None)
    t_solve = time.time() - t0
    print(f"  Solve time: {t_solve:.2f}s")

    print(f"\nStatus: {model.status}")
    print(f"Objective: €{model.objective.value:,.0f}")


if __name__ == '__main__':
    main()
