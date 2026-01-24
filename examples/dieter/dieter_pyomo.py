#!/usr/bin/env python3
"""
DIETER Energy System Model - Pyomo Implementation

Benchmark comparison with nimopt implementation.
"""

import argparse
import time
from pathlib import Path
import math

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Configuration
DATA_DIR = Path(__file__).parent / "data"
CURTAILMENT_COST = 0.1
INFEAS_COST = 10000


def load_data(n_hours: int = 8760):
    """Load all data from CSV files."""
    
    # Technologies
    tech_df = pd.read_csv(DATA_DIR / "technologies.csv")
    tech_params = pd.read_csv(DATA_DIR / "tech_params.csv").set_index('Technologies')
    
    dispatchable = tech_df[tech_df['Dispatchable'] == 1]['Technologies'].tolist()
    non_dispatchable = tech_df[tech_df['Dispatchable'] == 0]['Technologies'].tolist()
    
    # Storage
    sto_df = pd.read_csv(DATA_DIR / "sto_params.csv").set_index('Storages')
    storages = sto_df.index.tolist()
    
    # Hours
    hours = list(range(1, n_hours + 1))
    
    # Load
    load_df = pd.read_csv(DATA_DIR / "load.csv")
    load_data = {h: load_df.loc[h-1, 'Load'] for h in hours}
    
    # Availability (only non-dispatchable)
    avail_df = pd.read_csv(DATA_DIR / "availability.csv")
    avail_df = avail_df[
        (avail_df['Technologies'].isin(non_dispatchable)) & 
        (avail_df['Hours'] <= n_hours)
    ]
    availability = {
        (row['Technologies'], row['Hours']): row['Availability']
        for _, row in avail_df.iterrows()
    }
    
    return {
        'dispatchable': dispatchable,
        'non_dispatchable': non_dispatchable,
        'storages': storages,
        'hours': hours,
        'tech_params': tech_params,
        'sto_params': sto_df,
        'load': load_data,
        'availability': availability,
    }


def build_model(data: dict, corr_factor: float = 1.0):
    """Build the DIETER Pyomo model."""
    
    m = pyo.ConcreteModel()
    
    # Sets
    m.Dispatchable = pyo.Set(initialize=data['dispatchable'])
    m.NonDispatchable = pyo.Set(initialize=data['non_dispatchable'])
    m.Storages = pyo.Set(initialize=data['storages'])
    m.Hours = pyo.Set(initialize=data['hours'])
    
    tech_params = data['tech_params']
    sto_params = data['sto_params']
    
    # Parameters - Dispatchable
    m.InvestmentCost_D = pyo.Param(m.Dispatchable, 
        initialize={t: tech_params.loc[t, 'InvestmentCost'] for t in m.Dispatchable})
    m.MarginalCost_D = pyo.Param(m.Dispatchable,
        initialize={t: tech_params.loc[t, 'MarginalCost'] for t in m.Dispatchable})
    m.FixedCost_D = pyo.Param(m.Dispatchable,
        initialize={t: tech_params.loc[t, 'FixedCost'] for t in m.Dispatchable})
    
    # Parameters - NonDispatchable
    m.InvestmentCost_ND = pyo.Param(m.NonDispatchable,
        initialize={t: tech_params.loc[t, 'InvestmentCost'] for t in m.NonDispatchable})
    m.FixedCost_ND = pyo.Param(m.NonDispatchable,
        initialize={t: tech_params.loc[t, 'FixedCost'] for t in m.NonDispatchable})
    m.MaxInstallable_ND = pyo.Param(m.NonDispatchable,
        initialize={t: tech_params.loc[t, 'MaxInstallable'] for t in m.NonDispatchable})
    
    # Parameters - Storage
    m.InvestmentCostEnergy = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'InvestmentCostEnergy'] for s in m.Storages})
    m.InvestmentCostPower = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'InvestmentCostPower'] for s in m.Storages})
    m.MarginalCost_Sto = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'MarginalCost'] for s in m.Storages})
    m.FixedCost_Sto = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'FixedCost'] for s in m.Storages})
    m.Efficiency_Sto = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'Efficiency'] for s in m.Storages})
    m.MaxEnergy_Sto = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'MaxEnergy'] for s in m.Storages})
    m.MaxPower_Sto = pyo.Param(m.Storages,
        initialize={s: sto_params.loc[s, 'MaxPower'] for s in m.Storages})
    
    # Time series parameters
    m.Load = pyo.Param(m.Hours, initialize=data['load'])
    m.Availability = pyo.Param(m.NonDispatchable, m.Hours, 
        initialize=data['availability'], default=0)
    
    # Variables
    m.G_D = pyo.Var(m.Dispatchable, m.Hours, domain=pyo.NonNegativeReals)
    m.G_ND = pyo.Var(m.NonDispatchable, m.Hours, domain=pyo.NonNegativeReals)
    m.CU = pyo.Var(m.NonDispatchable, m.Hours, domain=pyo.NonNegativeReals)
    m.STO_IN = pyo.Var(m.Storages, m.Hours, domain=pyo.NonNegativeReals)
    m.STO_OUT = pyo.Var(m.Storages, m.Hours, domain=pyo.NonNegativeReals)
    m.STO_L = pyo.Var(m.Storages, m.Hours, domain=pyo.NonNegativeReals)
    m.N_D = pyo.Var(m.Dispatchable, domain=pyo.NonNegativeReals)
    m.N_ND = pyo.Var(m.NonDispatchable, domain=pyo.NonNegativeReals)
    m.N_STO_E = pyo.Var(m.Storages, domain=pyo.NonNegativeReals)
    m.N_STO_P = pyo.Var(m.Storages, domain=pyo.NonNegativeReals)
    m.G_INF = pyo.Var(m.Hours, domain=pyo.NonNegativeReals)
    
    # Objective
    def obj_rule(m):
        # Operating costs
        op_disp = sum(m.MarginalCost_D[t] * m.G_D[t, h] 
                      for t in m.Dispatchable for h in m.Hours)
        op_sto = sum(m.MarginalCost_Sto[s] * (m.STO_IN[s, h] + m.STO_OUT[s, h])
                     for s in m.Storages for h in m.Hours)
        cu_cost = CURTAILMENT_COST * sum(m.CU[t, h] 
                     for t in m.NonDispatchable for h in m.Hours)
        inf_cost = INFEAS_COST * sum(m.G_INF[h] for h in m.Hours)
        
        # Investment costs
        inv_d = corr_factor * sum(m.InvestmentCost_D[t] * m.N_D[t] for t in m.Dispatchable)
        inv_nd = corr_factor * sum(m.InvestmentCost_ND[t] * m.N_ND[t] for t in m.NonDispatchable)
        inv_sto_e = corr_factor * sum(m.InvestmentCostEnergy[s] * m.N_STO_E[s] for s in m.Storages)
        inv_sto_p = corr_factor * sum(m.InvestmentCostPower[s] * m.N_STO_P[s] for s in m.Storages)
        
        # Fixed costs
        fixed_d = corr_factor * sum(m.FixedCost_D[t] * m.N_D[t] for t in m.Dispatchable)
        fixed_nd = corr_factor * sum(m.FixedCost_ND[t] * m.N_ND[t] for t in m.NonDispatchable)
        fixed_sto = corr_factor * sum(m.FixedCost_Sto[s] * 0.5 * (m.N_STO_P[s] + m.N_STO_E[s]) 
                                       for s in m.Storages)
        
        return (op_disp + op_sto + cu_cost + inf_cost +
                inv_d + inv_nd + inv_sto_e + inv_sto_p +
                fixed_d + fixed_nd + fixed_sto)
    
    m.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Constraints
    
    # Energy Balance
    def energy_balance_rule(m, h):
        return (sum(m.G_D[t, h] for t in m.Dispatchable) +
                sum(m.G_ND[t, h] for t in m.NonDispatchable) +
                sum(m.STO_OUT[s, h] for s in m.Storages) +
                m.G_INF[h] ==
                sum(m.STO_IN[s, h] for s in m.Storages) + m.Load[h])
    m.EnergyBalance = pyo.Constraint(m.Hours, rule=energy_balance_rule)
    
    # Dispatchable generation limit
    def max_gen_disp_rule(m, t, h):
        return m.G_D[t, h] <= m.N_D[t]
    m.MaxGenDisp = pyo.Constraint(m.Dispatchable, m.Hours, rule=max_gen_disp_rule)
    
    # Non-dispatchable generation
    def max_gen_nd_rule(m, t, h):
        return m.G_ND[t, h] + m.CU[t, h] == m.Availability[t, h] * m.N_ND[t]
    m.MaxGenNonDisp = pyo.Constraint(m.NonDispatchable, m.Hours, rule=max_gen_nd_rule)
    
    # Maximum installable for non-dispatchable
    def max_install_nd_rule(m, t):
        return m.N_ND[t] <= m.MaxInstallable_ND[t]
    m.MaxInstallND = pyo.Constraint(m.NonDispatchable, rule=max_install_nd_rule)
    
    # Storage constraints
    def max_withdraw_rule(m, s, h):
        return m.STO_IN[s, h] <= m.N_STO_P[s]
    m.MaxWithdrawSto = pyo.Constraint(m.Storages, m.Hours, rule=max_withdraw_rule)
    
    def max_gen_sto_rule(m, s, h):
        return m.STO_OUT[s, h] <= m.N_STO_P[s]
    m.MaxGenSto = pyo.Constraint(m.Storages, m.Hours, rule=max_gen_sto_rule)
    
    def max_level_sto_rule(m, s, h):
        return m.STO_L[s, h] <= m.N_STO_E[s]
    m.MaxLevelSto = pyo.Constraint(m.Storages, m.Hours, rule=max_level_sto_rule)
    
    def max_energy_sto_rule(m, s):
        return m.N_STO_E[s] <= m.MaxEnergy_Sto[s]
    m.MaxEnergySto = pyo.Constraint(m.Storages, rule=max_energy_sto_rule)
    
    def max_power_sto_rule(m, s):
        return m.N_STO_P[s] <= m.MaxPower_Sto[s]
    m.MaxPowerSto = pyo.Constraint(m.Storages, rule=max_power_sto_rule)
    
    # Storage balance (cyclic)
    n_hours = len(data['hours'])
    def storage_balance_rule(m, s, h):
        sqrt_eff = math.sqrt(pyo.value(m.Efficiency_Sto[s]))
        h_prev = n_hours if h == 1 else h - 1
        return (m.STO_L[s, h] == m.STO_L[s, h_prev] + 
                sqrt_eff * m.STO_IN[s, h] - m.STO_OUT[s, h] / sqrt_eff)
    m.StorageBalance = pyo.Constraint(m.Storages, m.Hours, rule=storage_balance_rule)
    
    return m


def main():
    parser = argparse.ArgumentParser(description='DIETER - Pyomo')
    parser.add_argument('--hours', type=int, default=168)
    args = parser.parse_args()
    
    n_hours = args.hours
    corr_factor = n_hours / 8760
    
    print(f"\n{'='*60}")
    print(f"DIETER Energy System Model - Pyomo Implementation")
    print(f"{'='*60}")
    print(f"Hours: {n_hours}")
    
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
    solver = SolverFactory('appsi_highs')
    
    t0 = time.time()
    result = solver.solve(model, tee=False)
    t_solve = time.time() - t0
    print(f"  Total solve time: {t_solve:.2f}s")
    
    print(f"\nStatus: {result.solver.termination_condition}")
    print(f"Objective: €{pyo.value(model.objective):,.0f}")
    
    # Brief summary
    print("\nInstalled Capacities (MW):")
    for t in model.NonDispatchable:
        val = pyo.value(model.N_ND[t])
        if val > 0.1:
            print(f"  {t:20s}: {val:12.1f}")


if __name__ == '__main__':
    main()
