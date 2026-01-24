#!/usr/bin/env python3
"""
DIETER Benchmark - Compare nimopt, Pyomo, and linopy performance.

Measures:
1. Data loading time
2. Model building time  
3. Solver loading time (where separable)
4. Solve time
5. Total time
"""

import argparse
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CURTAILMENT_COST = 0.1
INFEAS_COST = 10000


def load_data(n_hours: int):
    """Load all data from CSV files - shared across all implementations."""
    tech_df = pd.read_csv(DATA_DIR / "technologies.csv")
    tech_params = pd.read_csv(DATA_DIR / "tech_params.csv").set_index('Technologies')
    sto_df = pd.read_csv(DATA_DIR / "sto_params.csv").set_index('Storages')
    load_df = pd.read_csv(DATA_DIR / "load.csv")
    avail_df = pd.read_csv(DATA_DIR / "availability.csv")
    
    dispatchable = tech_df[tech_df['Dispatchable'] == 1]['Technologies'].tolist()
    non_dispatchable = tech_df[tech_df['Dispatchable'] == 0]['Technologies'].tolist()
    storages = sto_df.index.tolist()
    hours = list(range(1, n_hours + 1))
    
    # Load time series
    load_data = {h: load_df.loc[h-1, 'Load'] for h in hours}
    
    # Availability - filter and create dict
    avail_filt = avail_df[
        (avail_df['Technologies'].isin(non_dispatchable)) & 
        (avail_df['Hours'] <= n_hours)
    ]
    availability = {
        (row['Technologies'], row['Hours']): row['Availability']
        for _, row in avail_filt.iterrows()
    }
    
    # Also create 2D numpy array for nimopt/linopy
    avail_matrix = np.zeros((len(non_dispatchable), n_hours))
    for i, t in enumerate(non_dispatchable):
        td = avail_filt[avail_filt['Technologies'] == t].sort_values('Hours')
        if len(td) == n_hours:
            avail_matrix[i, :] = td['Availability'].values
    
    return {
        'tech_df': tech_df,
        'tech_params': tech_params,
        'sto_params': sto_df,
        'dispatchable': dispatchable,
        'non_dispatchable': non_dispatchable,
        'storages': storages,
        'hours': hours,
        'load': load_data,
        'availability': availability,
        'avail_matrix': avail_matrix,
    }


# =============================================================================
# NIMOPT Implementation
# =============================================================================

def benchmark_nimopt(data: dict, corr_factor: float, use_rust: bool = True):
    """Benchmark nimopt implementation."""
    import nimopt as no
    from nimopt import sqrt, Sum
    from nimopt.solvers import HiGHSDirectSolver
    
    tech_params = data['tech_params']
    sto_params = data['sto_params']
    disp_list = data['dispatchable']
    nd_list = data['non_dispatchable']
    sto_list = data['storages']
    n_hours = len(data['hours'])
    
    # Build time
    t0 = time.time()
    
    # Sets
    Dispatchable = no.Set('Dispatchable', disp_list)
    NonDispatchable = no.Set('NonDispatchable', nd_list)
    Storages = no.Set('Storages', sto_list)
    Hours = no.Set('Hours', data['hours'])
    
    # Parameters
    InvCost_D = no.Param('InvCost_D', [Dispatchable], 
                         [tech_params.loc[t, 'InvestmentCost'] for t in disp_list])
    MargCost_D = no.Param('MargCost_D', [Dispatchable],
                          [tech_params.loc[t, 'MarginalCost'] for t in disp_list])
    FixCost_D = no.Param('FixCost_D', [Dispatchable],
                         [tech_params.loc[t, 'FixedCost'] for t in disp_list])
    InvCost_ND = no.Param('InvCost_ND', [NonDispatchable],
                          [tech_params.loc[t, 'InvestmentCost'] for t in nd_list])
    FixCost_ND = no.Param('FixCost_ND', [NonDispatchable],
                          [tech_params.loc[t, 'FixedCost'] for t in nd_list])
    MaxInst_ND = no.Param('MaxInst_ND', [NonDispatchable],
                          [tech_params.loc[t, 'MaxInstallable'] for t in nd_list])
    
    InvCostE = no.Param('InvCostE', [Storages],
                        [sto_params.loc[s, 'InvestmentCostEnergy'] for s in sto_list])
    InvCostP = no.Param('InvCostP', [Storages],
                        [sto_params.loc[s, 'InvestmentCostPower'] for s in sto_list])
    MargCost_S = no.Param('MargCost_S', [Storages],
                          [sto_params.loc[s, 'MarginalCost'] for s in sto_list])
    FixCost_S = no.Param('FixCost_S', [Storages],
                         [sto_params.loc[s, 'FixedCost'] for s in sto_list])
    Eff_S = no.Param('Eff_S', [Storages],
                     [sto_params.loc[s, 'Efficiency'] for s in sto_list])
    MaxE_S = no.Param('MaxE_S', [Storages],
                      [sto_params.loc[s, 'MaxEnergy'] for s in sto_list])
    MaxP_S = no.Param('MaxP_S', [Storages],
                      [sto_params.loc[s, 'MaxPower'] for s in sto_list])
    sqrt_eff = sqrt(Eff_S)
    
    Load = no.Param('Load', [Hours], 
                    [data['load'][h] for h in data['hours']])
    Availability = no.Param('Availability', [NonDispatchable, Hours], 
                            data['avail_matrix'])
    
    # Model
    m = no.Model(sense='minimize')
    
    # Variables
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
    op_d = Sum(Dispatchable, Hours, MargCost_D[Dispatchable] * G_D[Dispatchable, Hours])
    op_s = Sum(Storages, Hours, MargCost_S[Storages] * (STO_IN[Storages, Hours] + STO_OUT[Storages, Hours]))
    cu_cost = CURTAILMENT_COST * Sum(NonDispatchable, Hours, CU[NonDispatchable, Hours])
    inf_cost = INFEAS_COST * Sum(Hours, G_INF[Hours])
    
    inv_d = corr_factor * Sum(Dispatchable, InvCost_D[Dispatchable] * N_D[Dispatchable])
    inv_nd = corr_factor * Sum(NonDispatchable, InvCost_ND[NonDispatchable] * N_ND[NonDispatchable])
    inv_se = corr_factor * Sum(Storages, InvCostE[Storages] * N_STO_E[Storages])
    inv_sp = corr_factor * Sum(Storages, InvCostP[Storages] * N_STO_P[Storages])
    
    fix_d = corr_factor * Sum(Dispatchable, FixCost_D[Dispatchable] * N_D[Dispatchable])
    fix_nd = corr_factor * Sum(NonDispatchable, FixCost_ND[NonDispatchable] * N_ND[NonDispatchable])
    fix_s = corr_factor * Sum(Storages, FixCost_S[Storages] * 0.5 * (N_STO_P[Storages] + N_STO_E[Storages]))
    
    m.objective = (op_d + op_s + cu_cost + inf_cost + 
                   inv_d + inv_nd + inv_se + inv_sp +
                   fix_d + fix_nd + fix_s)
    
    # Constraints
    m.eq('EnergyBalance',
         Sum(Dispatchable, G_D[Dispatchable, Hours]) +
         Sum(NonDispatchable, G_ND[NonDispatchable, Hours]) +
         Sum(Storages, STO_OUT[Storages, Hours]) + G_INF[Hours] ==
         Sum(Storages, STO_IN[Storages, Hours]) + Load[Hours])
    
    m.eq('MaxGenDisp', G_D[Dispatchable, Hours] <= N_D[Dispatchable])
    m.eq('MaxGenND', G_ND[NonDispatchable, Hours] + CU[NonDispatchable, Hours] == 
         Availability[NonDispatchable, Hours] * N_ND[NonDispatchable])
    m.eq('MaxInstallND', N_ND[NonDispatchable] <= MaxInst_ND[NonDispatchable])
    
    m.eq('MaxWithdraw', STO_IN[Storages, Hours] <= N_STO_P[Storages])
    m.eq('MaxGenSto', STO_OUT[Storages, Hours] <= N_STO_P[Storages])
    m.eq('MaxLevel', STO_L[Storages, Hours] <= N_STO_E[Storages])
    m.eq('MaxEnergy', N_STO_E[Storages] <= MaxE_S[Storages])
    m.eq('MaxPower', N_STO_P[Storages] <= MaxP_S[Storages])
    
    m.eq('StorageBalance',
         STO_L[Storages, Hours] == 
         STO_L[Storages, Hours.lag(1, cyclic=True)] +
         sqrt_eff[Storages] * STO_IN[Storages, Hours] -
         STO_OUT[Storages, Hours] / sqrt_eff[Storages])
    
    t_build = time.time() - t0
    
    # Solver loading
    t0 = time.time()
    solver = HiGHSDirectSolver(use_rust=use_rust)
    solver.load_model(m)
    t_load = time.time() - t0
    
    # Solve
    t0 = time.time()
    result = solver.solve()
    t_solve = time.time() - t0
    
    return {
        'build': t_build,
        'load': t_load,
        'solve': t_solve,
        'total': t_build + t_load + t_solve,
        'objective': result.objective_value,
        'status': result.status.name,
    }


# =============================================================================
# PYOMO Implementation
# =============================================================================

def benchmark_pyomo(data: dict, corr_factor: float):
    """Benchmark Pyomo implementation."""
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    import math
    
    tech_params = data['tech_params']
    sto_params = data['sto_params']
    disp_list = data['dispatchable']
    nd_list = data['non_dispatchable']
    sto_list = data['storages']
    hours = data['hours']
    n_hours = len(hours)
    availability = data['availability']
    load_data = data['load']
    
    # Build time
    t0 = time.time()
    
    m = pyo.ConcreteModel()
    
    # Sets
    m.D = pyo.Set(initialize=disp_list)
    m.ND = pyo.Set(initialize=nd_list)
    m.S = pyo.Set(initialize=sto_list)
    m.H = pyo.Set(initialize=hours)
    
    # Parameters as dicts (faster than Pyomo Param for building)
    InvCost_D = {t: tech_params.loc[t, 'InvestmentCost'] for t in disp_list}
    MargCost_D = {t: tech_params.loc[t, 'MarginalCost'] for t in disp_list}
    FixCost_D = {t: tech_params.loc[t, 'FixedCost'] for t in disp_list}
    InvCost_ND = {t: tech_params.loc[t, 'InvestmentCost'] for t in nd_list}
    FixCost_ND = {t: tech_params.loc[t, 'FixedCost'] for t in nd_list}
    MaxInst_ND = {t: tech_params.loc[t, 'MaxInstallable'] for t in nd_list}
    
    InvCostE = {s: sto_params.loc[s, 'InvestmentCostEnergy'] for s in sto_list}
    InvCostP = {s: sto_params.loc[s, 'InvestmentCostPower'] for s in sto_list}
    MargCost_S = {s: sto_params.loc[s, 'MarginalCost'] for s in sto_list}
    FixCost_S = {s: sto_params.loc[s, 'FixedCost'] for s in sto_list}
    Eff_S = {s: sto_params.loc[s, 'Efficiency'] for s in sto_list}
    MaxE_S = {s: sto_params.loc[s, 'MaxEnergy'] for s in sto_list}
    MaxP_S = {s: sto_params.loc[s, 'MaxPower'] for s in sto_list}
    
    # Variables
    m.G_D = pyo.Var(m.D, m.H, domain=pyo.NonNegativeReals)
    m.G_ND = pyo.Var(m.ND, m.H, domain=pyo.NonNegativeReals)
    m.CU = pyo.Var(m.ND, m.H, domain=pyo.NonNegativeReals)
    m.STO_IN = pyo.Var(m.S, m.H, domain=pyo.NonNegativeReals)
    m.STO_OUT = pyo.Var(m.S, m.H, domain=pyo.NonNegativeReals)
    m.STO_L = pyo.Var(m.S, m.H, domain=pyo.NonNegativeReals)
    m.N_D = pyo.Var(m.D, domain=pyo.NonNegativeReals)
    m.N_ND = pyo.Var(m.ND, domain=pyo.NonNegativeReals)
    m.N_STO_E = pyo.Var(m.S, domain=pyo.NonNegativeReals)
    m.N_STO_P = pyo.Var(m.S, domain=pyo.NonNegativeReals)
    m.G_INF = pyo.Var(m.H, domain=pyo.NonNegativeReals)
    
    # Objective
    def obj_rule(m):
        op_d = sum(MargCost_D[t] * m.G_D[t, h] for t in disp_list for h in hours)
        op_s = sum(MargCost_S[s] * (m.STO_IN[s, h] + m.STO_OUT[s, h]) 
                   for s in sto_list for h in hours)
        cu = CURTAILMENT_COST * sum(m.CU[t, h] for t in nd_list for h in hours)
        inf = INFEAS_COST * sum(m.G_INF[h] for h in hours)
        
        inv_d = corr_factor * sum(InvCost_D[t] * m.N_D[t] for t in disp_list)
        inv_nd = corr_factor * sum(InvCost_ND[t] * m.N_ND[t] for t in nd_list)
        inv_se = corr_factor * sum(InvCostE[s] * m.N_STO_E[s] for s in sto_list)
        inv_sp = corr_factor * sum(InvCostP[s] * m.N_STO_P[s] for s in sto_list)
        
        fix_d = corr_factor * sum(FixCost_D[t] * m.N_D[t] for t in disp_list)
        fix_nd = corr_factor * sum(FixCost_ND[t] * m.N_ND[t] for t in nd_list)
        fix_s = corr_factor * sum(FixCost_S[s] * 0.5 * (m.N_STO_P[s] + m.N_STO_E[s]) 
                                   for s in sto_list)
        
        return (op_d + op_s + cu + inf + inv_d + inv_nd + inv_se + inv_sp +
                fix_d + fix_nd + fix_s)
    
    m.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Constraints
    def eb_rule(m, h):
        return (sum(m.G_D[t, h] for t in disp_list) +
                sum(m.G_ND[t, h] for t in nd_list) +
                sum(m.STO_OUT[s, h] for s in sto_list) + m.G_INF[h] ==
                sum(m.STO_IN[s, h] for s in sto_list) + load_data[h])
    m.EnergyBalance = pyo.Constraint(m.H, rule=eb_rule)
    
    def mgd_rule(m, t, h):
        return m.G_D[t, h] <= m.N_D[t]
    m.MaxGenDisp = pyo.Constraint(m.D, m.H, rule=mgd_rule)
    
    def mgnd_rule(m, t, h):
        return m.G_ND[t, h] + m.CU[t, h] == availability.get((t, h), 0) * m.N_ND[t]
    m.MaxGenND = pyo.Constraint(m.ND, m.H, rule=mgnd_rule)
    
    def mind_rule(m, t):
        return m.N_ND[t] <= MaxInst_ND[t]
    m.MaxInstallND = pyo.Constraint(m.ND, rule=mind_rule)
    
    def mw_rule(m, s, h):
        return m.STO_IN[s, h] <= m.N_STO_P[s]
    m.MaxWithdraw = pyo.Constraint(m.S, m.H, rule=mw_rule)
    
    def mg_rule(m, s, h):
        return m.STO_OUT[s, h] <= m.N_STO_P[s]
    m.MaxGenSto = pyo.Constraint(m.S, m.H, rule=mg_rule)
    
    def ml_rule(m, s, h):
        return m.STO_L[s, h] <= m.N_STO_E[s]
    m.MaxLevel = pyo.Constraint(m.S, m.H, rule=ml_rule)
    
    def me_rule(m, s):
        return m.N_STO_E[s] <= MaxE_S[s]
    m.MaxEnergy = pyo.Constraint(m.S, rule=me_rule)
    
    def mp_rule(m, s):
        return m.N_STO_P[s] <= MaxP_S[s]
    m.MaxPower = pyo.Constraint(m.S, rule=mp_rule)
    
    def sb_rule(m, s, h):
        sqrt_eff = math.sqrt(Eff_S[s])
        h_prev = n_hours if h == 1 else h - 1
        return (m.STO_L[s, h] == m.STO_L[s, h_prev] + 
                sqrt_eff * m.STO_IN[s, h] - m.STO_OUT[s, h] / sqrt_eff)
    m.StorageBalance = pyo.Constraint(m.S, m.H, rule=sb_rule)
    
    t_build = time.time() - t0
    
    # Solve (Pyomo combines load+solve)
    solver = SolverFactory('appsi_highs')
    
    t0 = time.time()
    result = solver.solve(m, tee=False)
    t_solve = time.time() - t0
    
    return {
        'build': t_build,
        'load': 0,  # Combined with solve for Pyomo
        'solve': t_solve,
        'total': t_build + t_solve,
        'objective': pyo.value(m.objective),
        'status': str(result.solver.termination_condition),
    }


# =============================================================================
# LINOPY Implementation  
# =============================================================================

def benchmark_linopy(data: dict, corr_factor: float):
    """Benchmark linopy implementation."""
    import linopy
    import xarray as xr
    
    tech_params = data['tech_params']
    sto_params = data['sto_params']
    disp_list = data['dispatchable']
    nd_list = data['non_dispatchable']
    sto_list = data['storages']
    hours = data['hours']
    n_hours = len(hours)
    
    # Build time
    t0 = time.time()
    
    m = linopy.Model()
    
    # Coordinates
    D = pd.Index(disp_list, name='D')
    ND = pd.Index(nd_list, name='ND')
    S = pd.Index(sto_list, name='S')
    H = pd.Index(hours, name='H')
    
    # Parameters as xarray DataArrays
    InvCost_D = xr.DataArray([tech_params.loc[t, 'InvestmentCost'] for t in disp_list], dims=['D'], coords={'D': D})
    MargCost_D = xr.DataArray([tech_params.loc[t, 'MarginalCost'] for t in disp_list], dims=['D'], coords={'D': D})
    FixCost_D = xr.DataArray([tech_params.loc[t, 'FixedCost'] for t in disp_list], dims=['D'], coords={'D': D})
    
    InvCost_ND = xr.DataArray([tech_params.loc[t, 'InvestmentCost'] for t in nd_list], dims=['ND'], coords={'ND': ND})
    FixCost_ND = xr.DataArray([tech_params.loc[t, 'FixedCost'] for t in nd_list], dims=['ND'], coords={'ND': ND})
    MaxInst_ND = xr.DataArray([tech_params.loc[t, 'MaxInstallable'] for t in nd_list], dims=['ND'], coords={'ND': ND})
    
    InvCostE = xr.DataArray([sto_params.loc[s, 'InvestmentCostEnergy'] for s in sto_list], dims=['S'], coords={'S': S})
    InvCostP = xr.DataArray([sto_params.loc[s, 'InvestmentCostPower'] for s in sto_list], dims=['S'], coords={'S': S})
    MargCost_S = xr.DataArray([sto_params.loc[s, 'MarginalCost'] for s in sto_list], dims=['S'], coords={'S': S})
    FixCost_S = xr.DataArray([sto_params.loc[s, 'FixedCost'] for s in sto_list], dims=['S'], coords={'S': S})
    Eff_S = xr.DataArray([sto_params.loc[s, 'Efficiency'] for s in sto_list], dims=['S'], coords={'S': S})
    MaxE_S = xr.DataArray([sto_params.loc[s, 'MaxEnergy'] for s in sto_list], dims=['S'], coords={'S': S})
    MaxP_S = xr.DataArray([sto_params.loc[s, 'MaxPower'] for s in sto_list], dims=['S'], coords={'S': S})
    sqrt_eff = np.sqrt(Eff_S)
    
    Load = xr.DataArray([data['load'][h] for h in hours], dims=['H'], coords={'H': H})
    Availability = xr.DataArray(data['avail_matrix'], dims=['ND', 'H'], coords={'ND': ND, 'H': H})
    
    # Variables
    G_D = m.add_variables(lower=0, coords=[D, H], name='G_D')
    G_ND = m.add_variables(lower=0, coords=[ND, H], name='G_ND')
    CU = m.add_variables(lower=0, coords=[ND, H], name='CU')
    STO_IN = m.add_variables(lower=0, coords=[S, H], name='STO_IN')
    STO_OUT = m.add_variables(lower=0, coords=[S, H], name='STO_OUT')
    STO_L = m.add_variables(lower=0, coords=[S, H], name='STO_L')
    N_D = m.add_variables(lower=0, coords=[D], name='N_D')
    N_ND = m.add_variables(lower=0, coords=[ND], name='N_ND')
    N_STO_E = m.add_variables(lower=0, coords=[S], name='N_STO_E')
    N_STO_P = m.add_variables(lower=0, coords=[S], name='N_STO_P')
    G_INF = m.add_variables(lower=0, coords=[H], name='G_INF')
    
    # Objective
    obj = (
        (MargCost_D * G_D).sum() +
        (MargCost_S * (STO_IN + STO_OUT)).sum() +
        CURTAILMENT_COST * CU.sum() +
        INFEAS_COST * G_INF.sum() +
        corr_factor * (InvCost_D * N_D).sum() +
        corr_factor * (InvCost_ND * N_ND).sum() +
        corr_factor * (InvCostE * N_STO_E).sum() +
        corr_factor * (InvCostP * N_STO_P).sum() +
        corr_factor * (FixCost_D * N_D).sum() +
        corr_factor * (FixCost_ND * N_ND).sum() +
        corr_factor * (FixCost_S * 0.5 * (N_STO_P + N_STO_E)).sum()
    )
    m.add_objective(obj, sense='min')
    
    # Constraints
    m.add_constraints(
        G_D.sum('D') + G_ND.sum('ND') + STO_OUT.sum('S') + G_INF == 
        STO_IN.sum('S') + Load,
        name='EnergyBalance'
    )
    
    m.add_constraints(G_D <= N_D, name='MaxGenDisp')
    m.add_constraints(G_ND + CU == Availability * N_ND, name='MaxGenND')
    m.add_constraints(N_ND <= MaxInst_ND, name='MaxInstallND')
    
    m.add_constraints(STO_IN <= N_STO_P, name='MaxWithdraw')
    m.add_constraints(STO_OUT <= N_STO_P, name='MaxGenSto')
    m.add_constraints(STO_L <= N_STO_E, name='MaxLevel')
    m.add_constraints(N_STO_E <= MaxE_S, name='MaxEnergy')
    m.add_constraints(N_STO_P <= MaxP_S, name='MaxPower')
    
    # Storage balance with cyclic shift
    STO_L_prev = STO_L.roll(H=1)  # Cyclic shift
    m.add_constraints(
        STO_L == STO_L_prev + sqrt_eff * STO_IN - STO_OUT / sqrt_eff,
        name='StorageBalance'
    )
    
    t_build = time.time() - t0
    
    # Solve
    t0 = time.time()
    m.solve(solver_name='highs', log_fn=None)
    t_solve = time.time() - t0
    
    return {
        'build': t_build,
        'load': 0,  # Combined with solve
        'solve': t_solve,
        'total': t_build + t_solve,
        'objective': m.objective.value,
        'status': m.status,
    }


# =============================================================================
# Main
# =============================================================================

def run_benchmark(n_hours: int, frameworks: list = None):
    """Run benchmark for specified frameworks."""
    if frameworks is None:
        frameworks = ['nimopt', 'pyomo', 'linopy']
    
    corr_factor = n_hours / 8760
    
    print(f"\n{'='*70}")
    print(f"DIETER Benchmark - {n_hours} hours ({n_hours/24:.1f} days)")
    print(f"{'='*70}")
    
    # Load data once
    print("\nLoading data...")
    t0 = time.time()
    data = load_data(n_hours)
    t_data = time.time() - t0
    print(f"  Data loading: {t_data:.3f}s")
    
    results = {}
    
    # Run each framework
    for fw in frameworks:
        gc.collect()
        print(f"\n--- {fw.upper()} ---")
        
        try:
            if fw == 'nimopt':
                r = benchmark_nimopt(data, corr_factor)
            elif fw == 'pyomo':
                r = benchmark_pyomo(data, corr_factor)
            elif fw == 'linopy':
                r = benchmark_linopy(data, corr_factor)
            else:
                continue
            
            results[fw] = r
            print(f"  Build:     {r['build']:.3f}s")
            if r['load'] > 0:
                print(f"  Load:      {r['load']:.3f}s")
            print(f"  Solve:     {r['solve']:.3f}s")
            print(f"  Total:     {r['total']:.3f}s")
            print(f"  Objective: €{r['objective']:,.0f}")
            print(f"  Status:    {r['status']}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[fw] = {'error': str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description='DIETER Benchmark')
    parser.add_argument('--hours', type=int, nargs='+', default=[168],
                        help='Hours to benchmark (can specify multiple)')
    parser.add_argument('--frameworks', type=str, nargs='+', 
                        default=['nimopt', 'pyomo', 'linopy'],
                        help='Frameworks to benchmark')
    args = parser.parse_args()
    
    all_results = {}
    for n_hours in args.hours:
        all_results[n_hours] = run_benchmark(n_hours, args.frameworks)
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Hours':<8} {'Framework':<10} {'Build':<10} {'Load':<10} {'Solve':<10} {'Total':<10}")
    print("-" * 70)
    
    for n_hours, results in all_results.items():
        for fw, r in results.items():
            if 'error' not in r:
                load_str = f"{r['load']:.3f}s" if r['load'] > 0 else "-"
                print(f"{n_hours:<8} {fw:<10} {r['build']:.3f}s     {load_str:<10} {r['solve']:.3f}s     {r['total']:.3f}s")


if __name__ == '__main__':
    main()
