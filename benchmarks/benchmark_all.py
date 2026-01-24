#!/usr/bin/env python3
"""
Unified Benchmark: nimopt vs Pyomo vs linopy

Compares DIETER model building and solving times across frameworks.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "examples" / "dieter" / "data"
CURTAILMENT_COST = 0.1
INFEAS_COST = 10000


def load_shared_data(n_hours: int):
    """Load data once for all frameworks."""
    tech_df = pd.read_csv(DATA_DIR / "technologies.csv")
    tech_params = pd.read_csv(DATA_DIR / "tech_params.csv").set_index('Technologies')
    sto_params = pd.read_csv(DATA_DIR / "sto_params.csv").set_index('Storages')
    load_df = pd.read_csv(DATA_DIR / "load.csv")
    avail_df = pd.read_csv(DATA_DIR / "availability.csv")

    dispatchable = tech_df[tech_df['Dispatchable'] == 1]['Technologies'].tolist()
    non_dispatchable = tech_df[tech_df['Dispatchable'] == 0]['Technologies'].tolist()
    storages = sto_params.index.tolist()
    hours = list(range(1, n_hours + 1))

    avail_df = avail_df[
        (avail_df['Technologies'].isin(non_dispatchable)) &
        (avail_df['Hours'] <= n_hours)
    ]
    avail_matrix = np.zeros((len(non_dispatchable), n_hours))
    for i, t in enumerate(non_dispatchable):
        tech_data = avail_df[avail_df['Technologies'] == t].sort_values('Hours')
        avail_matrix[i, :] = tech_data['Availability'].values

    return {
        'dispatchable': dispatchable,
        'non_dispatchable': non_dispatchable,
        'storages': storages,
        'hours': hours,
        'tech_params': tech_params,
        'sto_params': sto_params,
        'load': load_df.loc[:n_hours-1, 'Load'].values,
        'availability': avail_matrix,
    }


# ============================================================================
# nimopt implementation
# ============================================================================

def benchmark_nimopt(data: dict, corr_factor: float):
    """Run nimopt benchmark, return timings dict."""
    import nimopt as no
    from nimopt import sqrt, Sum
    from nimopt.solvers import HiGHSDirectSolver

    disp, nd, sto, hours = data['dispatchable'], data['non_dispatchable'], data['storages'], data['hours']
    tech_params, sto_params = data['tech_params'], data['sto_params']

    t0 = time.time()
    # Sets
    Dispatchable = no.Set('Dispatchable', disp)
    NonDispatchable = no.Set('NonDispatchable', nd)
    Storages = no.Set('Storages', sto)
    Hours = no.Set('Hours', hours)

    # Parameters
    inv_cost_disp = no.Param('InvestmentCost_Disp', [Dispatchable], 
                             [tech_params.loc[t, 'InvestmentCost'] for t in disp])
    inv_cost_nd = no.Param('InvestmentCost_ND', [NonDispatchable],
                           [tech_params.loc[t, 'InvestmentCost'] for t in nd])
    marg_cost_disp = no.Param('MarginalCost_Disp', [Dispatchable],
                              [tech_params.loc[t, 'MarginalCost'] for t in disp])
    fixed_cost_disp = no.Param('FixedCost_Disp', [Dispatchable],
                               [tech_params.loc[t, 'FixedCost'] for t in disp])
    fixed_cost_nd = no.Param('FixedCost_ND', [NonDispatchable],
                             [tech_params.loc[t, 'FixedCost'] for t in nd])
    max_install_nd = no.Param('MaxInstallable_ND', [NonDispatchable],
                              [tech_params.loc[t, 'MaxInstallable'] for t in nd])

    inv_cost_energy = no.Param('InvestmentCostEnergy', [Storages],
                               [sto_params.loc[s, 'InvestmentCostEnergy'] for s in sto])
    inv_cost_power = no.Param('InvestmentCostPower', [Storages],
                              [sto_params.loc[s, 'InvestmentCostPower'] for s in sto])
    marg_cost_sto = no.Param('MarginalCost_Sto', [Storages],
                             [sto_params.loc[s, 'MarginalCost'] for s in sto])
    fixed_cost_sto = no.Param('FixedCost_Sto', [Storages],
                              [sto_params.loc[s, 'FixedCost'] for s in sto])

    efficiency_sto = no.Param('Efficiency_Sto', [Storages],
                              [sto_params.loc[s, 'Efficiency'] for s in sto])
    max_energy_sto = no.Param('MaxEnergy_Sto', [Storages],
                              [sto_params.loc[s, 'MaxEnergy'] for s in sto])
    max_power_sto = no.Param('MaxPower_Sto', [Storages],
                             [sto_params.loc[s, 'MaxPower'] for s in sto])
    Load = no.Param('Load', [Hours], data['load'])
    Availability = no.Param('Availability', [NonDispatchable, Hours], data['availability'])
    sqrt_eff = sqrt(efficiency_sto)

    # Model & Variables
    m = no.Model(sense='minimize')
    G_D = m.var('G_D', [Dispatchable, Hours], lb=0)
    G_ND = m.var('G_ND', [NonDispatchable, Hours], lb=0)
    CU = m.var('CU', [NonDispatchable, Hours], lb=0)
    STO_IN = m.var('STO_IN', [Storages, Hours], lb=0)
    STO_OUT = m.var('STO_OUT', [Storages, Hours], lb=0)
    STO_L = m.var('STO_L', [Storages, Hours], lb=0)
    N_D = m.var('N_D', [Dispatchable], lb=0)
    N_ND = m.var('N_ND', [NonDispatchable], lb=0)
    N_STO_E = m.var('N_STO_E', [Storages], lb=0)
    N_STO_P = m.var('N_STO_P', [Storages], lb=0)
    G_INF = m.var('G_INF', [Hours], lb=0)

    # Objective
    m.objective = (
        Sum(Dispatchable, Hours, marg_cost_disp[Dispatchable] * G_D[Dispatchable, Hours]) +
        Sum(Storages, Hours, marg_cost_sto[Storages] * (STO_IN[Storages, Hours] + STO_OUT[Storages, Hours])) +
        CURTAILMENT_COST * Sum(NonDispatchable, Hours, CU[NonDispatchable, Hours]) +
        INFEAS_COST * Sum(Hours, G_INF[Hours]) +
        corr_factor * Sum(Dispatchable, inv_cost_disp[Dispatchable] * N_D[Dispatchable]) +
        corr_factor * Sum(NonDispatchable, inv_cost_nd[NonDispatchable] * N_ND[NonDispatchable]) +
        corr_factor * Sum(Storages, inv_cost_energy[Storages] * N_STO_E[Storages]) +
        corr_factor * Sum(Storages, inv_cost_power[Storages] * N_STO_P[Storages]) +

        corr_factor * Sum(Dispatchable, fixed_cost_disp[Dispatchable] * N_D[Dispatchable]) +
        corr_factor * Sum(NonDispatchable, fixed_cost_nd[NonDispatchable] * N_ND[NonDispatchable]) +
        corr_factor * Sum(Storages, fixed_cost_sto[Storages] * 0.5 * (N_STO_P[Storages] + N_STO_E[Storages]))
    )

    # Constraints
    m.eq('EnergyBalance',
         Sum(Dispatchable, G_D[Dispatchable, Hours]) +
         Sum(NonDispatchable, G_ND[NonDispatchable, Hours]) +
         Sum(Storages, STO_OUT[Storages, Hours]) + G_INF[Hours] ==
         Sum(Storages, STO_IN[Storages, Hours]) + Load[Hours])
    m.eq('MaxGenDisp', G_D[Dispatchable, Hours] <= N_D[Dispatchable])
    m.eq('MaxGenNonDisp', G_ND[NonDispatchable, Hours] + CU[NonDispatchable, Hours] == 
         Availability[NonDispatchable, Hours] * N_ND[NonDispatchable])
    m.eq('MaxInstallND', N_ND[NonDispatchable] <= max_install_nd[NonDispatchable])
    m.eq('MaxWithdrawSto', STO_IN[Storages, Hours] <= N_STO_P[Storages])
    m.eq('MaxGenSto', STO_OUT[Storages, Hours] <= N_STO_P[Storages])
    m.eq('MaxLevelSto', STO_L[Storages, Hours] <= N_STO_E[Storages])
    m.eq('MaxEnergySto', N_STO_E[Storages] <= max_energy_sto[Storages])
    m.eq('MaxPowerSto', N_STO_P[Storages] <= max_power_sto[Storages])
    m.eq('StorageBalance',
         STO_L[Storages, Hours] == STO_L[Storages, Hours.lag(1, cyclic=True)] +
         sqrt_eff[Storages] * STO_IN[Storages, Hours] -
         STO_OUT[Storages, Hours] / sqrt_eff[Storages])

    t_build = time.time() - t0

    # Solve
    t0 = time.time()
    solver = HiGHSDirectSolver(use_rust=True)
    solver.load_model(m)
    t_load = time.time() - t0

    t0 = time.time()
    result = solver.solve()
    t_solve = time.time() - t0

    return {'build': t_build, 'load': t_load, 'solve': t_solve, 
            'objective': result.objective_value, 'status': result.status.name}


# ============================================================================
# Pyomo implementation
# ============================================================================

def benchmark_pyomo(data: dict, corr_factor: float):
    """Run Pyomo benchmark, return timings dict."""
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory

    disp, nd, sto, hours = data['dispatchable'], data['non_dispatchable'], data['storages'], data['hours']
    tech_params, sto_params = data['tech_params'], data['sto_params']
    load_data = data['load']
    avail_matrix = data['availability']
    nd_to_idx = {t: i for i, t in enumerate(nd)}

    t0 = time.time()
    m = pyo.ConcreteModel()
    m.Dispatchable = pyo.Set(initialize=disp)
    m.NonDispatchable = pyo.Set(initialize=nd)
    m.Storages = pyo.Set(initialize=sto)
    m.Hours = pyo.Set(initialize=hours)

    # Parameters (use dict init for speed)
    m.InvestmentCost_Disp = pyo.Param(m.Dispatchable, 
        initialize={t: tech_params.loc[t, 'InvestmentCost'] for t in disp})
    m.InvestmentCost_ND = pyo.Param(m.NonDispatchable,
        initialize={t: tech_params.loc[t, 'InvestmentCost'] for t in nd})
    m.MarginalCost_Disp = pyo.Param(m.Dispatchable,
        initialize={t: tech_params.loc[t, 'MarginalCost'] for t in disp})
    m.FixedCost_Disp = pyo.Param(m.Dispatchable,
        initialize={t: tech_params.loc[t, 'FixedCost'] for t in disp})
    m.FixedCost_ND = pyo.Param(m.NonDispatchable,
        initialize={t: tech_params.loc[t, 'FixedCost'] for t in nd})
    m.MaxInstallable_ND = pyo.Param(m.NonDispatchable,
        initialize={t: tech_params.loc[t, 'MaxInstallable'] for t in nd})

    m.InvestmentCostEnergy = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'InvestmentCostEnergy'] for s in sto})
    m.InvestmentCostPower = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'InvestmentCostPower'] for s in sto})
    m.MarginalCost_Sto = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'MarginalCost'] for s in sto})
    m.FixedCost_Sto = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'FixedCost'] for s in sto})
    m.Efficiency_Sto = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'Efficiency'] for s in sto})
    m.MaxEnergy_Sto = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'MaxEnergy'] for s in sto})
    m.MaxPower_Sto = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'MaxPower'] for s in sto})
    m.Load = pyo.Param(m.Hours, initialize={h: float(load_data[h-1]) for h in hours})
    m.Availability = pyo.Param(m.NonDispatchable, m.Hours,
        initialize={(t, h): float(avail_matrix[nd_to_idx[t], h-1]) for t in nd for h in hours})
    m.CorrFactor = pyo.Param(initialize=corr_factor)

    # Variables
    m.G_D = pyo.Var(m.Dispatchable, m.Hours, within=pyo.NonNegativeReals)
    m.G_ND = pyo.Var(m.NonDispatchable, m.Hours, within=pyo.NonNegativeReals)
    m.CU = pyo.Var(m.NonDispatchable, m.Hours, within=pyo.NonNegativeReals)
    m.STO_IN = pyo.Var(m.Storages, m.Hours, within=pyo.NonNegativeReals)
    m.STO_OUT = pyo.Var(m.Storages, m.Hours, within=pyo.NonNegativeReals)
    m.STO_L = pyo.Var(m.Storages, m.Hours, within=pyo.NonNegativeReals)
    m.N_D = pyo.Var(m.Dispatchable, within=pyo.NonNegativeReals)
    m.N_ND = pyo.Var(m.NonDispatchable, within=pyo.NonNegativeReals)
    m.N_STO_E = pyo.Var(m.Storages, within=pyo.NonNegativeReals)
    m.N_STO_P = pyo.Var(m.Storages, within=pyo.NonNegativeReals)
    m.G_INF = pyo.Var(m.Hours, within=pyo.NonNegativeReals)

    # Objective
    def objective_rule(m):
        cf = m.CorrFactor
        return (sum(m.MarginalCost_Disp[t] * m.G_D[t, h] for t in m.Dispatchable for h in m.Hours) +
                sum(m.MarginalCost_Sto[s] * (m.STO_IN[s, h] + m.STO_OUT[s, h]) for s in m.Storages for h in m.Hours) +
                CURTAILMENT_COST * sum(m.CU[t, h] for t in m.NonDispatchable for h in m.Hours) +
                INFEAS_COST * sum(m.G_INF[h] for h in m.Hours) +
                cf * sum(m.InvestmentCost_Disp[t] * m.N_D[t] for t in m.Dispatchable) +
                cf * sum(m.InvestmentCost_ND[t] * m.N_ND[t] for t in m.NonDispatchable) +
                cf * sum(m.InvestmentCostEnergy[s] * m.N_STO_E[s] for s in m.Storages) +
                cf * sum(m.InvestmentCostPower[s] * m.N_STO_P[s] for s in m.Storages) +
                cf * sum(m.FixedCost_Disp[t] * m.N_D[t] for t in m.Dispatchable) +
                cf * sum(m.FixedCost_ND[t] * m.N_ND[t] for t in m.NonDispatchable) +
                cf * sum(m.FixedCost_Sto[s] * 0.5 * (m.N_STO_P[s] + m.N_STO_E[s]) for s in m.Storages))
    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    def energy_balance_rule(m, h):
        return (sum(m.G_D[t, h] for t in m.Dispatchable) +
                sum(m.G_ND[t, h] for t in m.NonDispatchable) +
                sum(m.STO_OUT[s, h] for s in m.Storages) + m.G_INF[h] ==
                sum(m.STO_IN[s, h] for s in m.Storages) + m.Load[h])
    m.EnergyBalance = pyo.Constraint(m.Hours, rule=energy_balance_rule)

    def max_gen_disp_rule(m, t, h):
        return m.G_D[t, h] <= m.N_D[t]
    m.MaxGenDisp = pyo.Constraint(m.Dispatchable, m.Hours, rule=max_gen_disp_rule)

    def max_gen_nd_rule(m, t, h):
        return m.G_ND[t, h] + m.CU[t, h] == m.Availability[t, h] * m.N_ND[t]
    m.MaxGenNonDisp = pyo.Constraint(m.NonDispatchable, m.Hours, rule=max_gen_nd_rule)

    def max_install_nd_rule(m, t):
        return m.N_ND[t] <= m.MaxInstallable_ND[t]
    m.MaxInstallND = pyo.Constraint(m.NonDispatchable, rule=max_install_nd_rule)

    m.MaxWithdrawSto = pyo.Constraint(m.Storages, m.Hours,
        rule=lambda m, s, h: m.STO_IN[s, h] <= m.N_STO_P[s])
    m.MaxGenSto = pyo.Constraint(m.Storages, m.Hours,
        rule=lambda m, s, h: m.STO_OUT[s, h] <= m.N_STO_P[s])
    m.MaxLevelSto = pyo.Constraint(m.Storages, m.Hours,
        rule=lambda m, s, h: m.STO_L[s, h] <= m.N_STO_E[s])
    m.MaxEnergySto = pyo.Constraint(m.Storages,
        rule=lambda m, s: m.N_STO_E[s] <= m.MaxEnergy_Sto[s])
    m.MaxPowerSto = pyo.Constraint(m.Storages,
        rule=lambda m, s: m.N_STO_P[s] <= m.MaxPower_Sto[s])

    # Storage balance
    hours_list = list(hours)
    def storage_balance_rule(m, s, h):
        sqrt_eff = pyo.sqrt(m.Efficiency_Sto[s])
        h_prev = hours_list[-1] if h == hours_list[0] else hours_list[hours_list.index(h) - 1]
        return m.STO_L[s, h] == m.STO_L[s, h_prev] + sqrt_eff * m.STO_IN[s, h] - m.STO_OUT[s, h] / sqrt_eff
    m.StorageBalance = pyo.Constraint(m.Storages, m.Hours, rule=storage_balance_rule)

    t_build = time.time() - t0

    # Solve
    t0 = time.time()
    solver = SolverFactory('appsi_highs')
    result = solver.solve(m, tee=False)
    t_solve = time.time() - t0

    return {'build': t_build, 'load': 0.0, 'solve': t_solve,
            'objective': pyo.value(m.objective), 'status': str(result.solver.termination_condition)}


# ============================================================================
# linopy implementation
# ============================================================================

def benchmark_linopy(data: dict, corr_factor: float):
    """Run linopy benchmark, return timings dict."""
    import xarray as xr
    import linopy

    disp, nd, sto, hours = data['dispatchable'], data['non_dispatchable'], data['storages'], data['hours']
    tech_params, sto_params = data['tech_params'], data['sto_params']

    t0 = time.time()
    m = linopy.Model()

    # Parameters as xarray
    load = xr.DataArray(data['load'], dims=['Hours'], coords={'Hours': hours})
    avail = xr.DataArray(data['availability'], dims=['NonDispatchable', 'Hours'],
                         coords={'NonDispatchable': nd, 'Hours': hours})
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
    G_D = m.add_variables(lower=0, dims=['Dispatchable', 'Hours'], coords=[disp, hours], name='G_D')
    G_ND = m.add_variables(lower=0, dims=['NonDispatchable', 'Hours'], coords=[nd, hours], name='G_ND')
    CU = m.add_variables(lower=0, dims=['NonDispatchable', 'Hours'], coords=[nd, hours], name='CU')
    STO_IN = m.add_variables(lower=0, dims=['Storages', 'Hours'], coords=[sto, hours], name='STO_IN')
    STO_OUT = m.add_variables(lower=0, dims=['Storages', 'Hours'], coords=[sto, hours], name='STO_OUT')
    STO_L = m.add_variables(lower=0, dims=['Storages', 'Hours'], coords=[sto, hours], name='STO_L')
    N_D = m.add_variables(lower=0, dims=['Dispatchable'], coords=[disp], name='N_D')
    N_ND = m.add_variables(lower=0, dims=['NonDispatchable'], coords=[nd], name='N_ND')
    N_STO_E = m.add_variables(lower=0, dims=['Storages'], coords=[sto], name='N_STO_E')
    N_STO_P = m.add_variables(lower=0, dims=['Storages'], coords=[sto], name='N_STO_P')
    G_INF = m.add_variables(lower=0, dims=['Hours'], coords=[hours], name='G_INF')

    # Objective
    obj = ((marg_cost_disp * G_D).sum() + (marg_cost_sto * (STO_IN + STO_OUT)).sum() +
           CURTAILMENT_COST * CU.sum() + INFEAS_COST * G_INF.sum() +
           corr_factor * (inv_cost_disp * N_D).sum() + corr_factor * (inv_cost_nd * N_ND).sum() +
           corr_factor * (inv_cost_energy * N_STO_E).sum() + corr_factor * (inv_cost_power * N_STO_P).sum() +
           corr_factor * (fixed_cost_disp * N_D).sum() + corr_factor * (fixed_cost_nd * N_ND).sum() +
           corr_factor * (fixed_cost_sto * 0.5 * (N_STO_P + N_STO_E)).sum())
    m.add_objective(obj, sense='min')

    # Constraints
    m.add_constraints(G_D.sum('Dispatchable') + G_ND.sum('NonDispatchable') + 
                      STO_OUT.sum('Storages') + G_INF == STO_IN.sum('Storages') + load,
                      name='EnergyBalance')
    m.add_constraints(G_D <= N_D, name='MaxGenDisp')
    m.add_constraints(G_ND + CU == avail * N_ND, name='MaxGenNonDisp')
    m.add_constraints(N_ND <= max_install_nd, name='MaxInstallND')
    m.add_constraints(STO_IN <= N_STO_P, name='MaxWithdrawSto')
    m.add_constraints(STO_OUT <= N_STO_P, name='MaxGenSto')
    m.add_constraints(STO_L <= N_STO_E, name='MaxLevelSto')
    m.add_constraints(N_STO_E <= max_energy_sto, name='MaxEnergySto')
    m.add_constraints(N_STO_P <= max_power_sto, name='MaxPowerSto')
    m.add_constraints(STO_L == STO_L.roll(Hours=1) + sqrt_eff * STO_IN - STO_OUT / sqrt_eff,
                      name='StorageBalance')

    t_build = time.time() - t0

    # Solve
    t0 = time.time()
    m.solve(solver_name='highs', log_fn=None)
    t_solve = time.time() - t0

    return {'build': t_build, 'load': 0.0, 'solve': t_solve,
            'objective': m.objective.value, 'status': m.status}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark nimopt vs Pyomo vs linopy')
    parser.add_argument('--hours', type=int, nargs='+', default=[168, 720, 2000, 8760],
                        help='Hour scales to test')
    parser.add_argument('--frameworks', type=str, nargs='+', 
                        default=['nimopt', 'pyomo', 'linopy'],
                        help='Frameworks to benchmark')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("DIETER Benchmark: nimopt vs Pyomo vs linopy")
    print("=" * 70)

    results = []
    for n_hours in args.hours:
        corr_factor = n_hours / 8760
        print(f"\n--- {n_hours} hours ({n_hours/24:.1f} days) ---")

        # Load data once
        t0 = time.time()
        data = load_shared_data(n_hours)
        t_data = time.time() - t0
        print(f"Data loading: {t_data:.3f}s")

        for framework in args.frameworks:
            print(f"\n{framework}:")
            try:
                if framework == 'nimopt':
                    r = benchmark_nimopt(data, corr_factor)
                elif framework == 'pyomo':
                    r = benchmark_pyomo(data, corr_factor)
                elif framework == 'linopy':
                    r = benchmark_linopy(data, corr_factor)
                else:
                    print(f"  Unknown framework: {framework}")
                    continue

                total = r['build'] + r['load'] + r['solve']
                print(f"  Build:  {r['build']:.3f}s")
                if r['load'] > 0:
                    print(f"  Load:   {r['load']:.3f}s")
                print(f"  Solve:  {r['solve']:.3f}s")
                print(f"  Total:  {total:.3f}s")
                print(f"  Obj:    €{r['objective']:,.0f}")

                results.append({
                    'hours': n_hours, 'framework': framework,
                    'build': r['build'], 'load': r['load'], 'solve': r['solve'],
                    'total': total, 'objective': r['objective'], 'status': r['status']
                })
            except Exception as e:
                print(f"  ERROR: {e}")

    # Summary table
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        df = pd.DataFrame(results)
        
        # Pivot for comparison
        for metric in ['build', 'solve', 'total']:
            print(f"\n{metric.upper()} TIME (seconds):")
            pivot = df.pivot(index='hours', columns='framework', values=metric)
            print(pivot.to_string())


if __name__ == '__main__':
    main()
