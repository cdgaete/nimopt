#!/usr/bin/env python3
"""
DIETER Energy System Model - Pyomo Implementation

For benchmarking comparison with nimopt.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

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
    all_tech = tech_df['Technologies'].tolist()
    dispatchable = tech_df[tech_df['Dispatchable'] == 1]['Technologies'].tolist()
    non_dispatchable = tech_df[tech_df['Dispatchable'] == 0]['Technologies'].tolist()
    storages = sto_params.index.tolist()
    hours = list(range(1, n_hours + 1))

    # Filter availability to relevant techs and hours
    avail_df = avail_df[
        (avail_df['Technologies'].isin(non_dispatchable)) &
        (avail_df['Hours'] <= n_hours)
    ]

    return {
        'all_tech': all_tech,
        'dispatchable': dispatchable,
        'non_dispatchable': non_dispatchable,
        'storages': storages,
        'hours': hours,
        'tech_params': tech_params,
        'sto_params': sto_params,
        'load': load_df.loc[:n_hours-1, 'Load'].values,
        'availability': avail_df,
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

    # Parameters
    def investment_cost_disp_init(m, t):
        return tech_params.loc[t, 'InvestmentCost']
    m.InvestmentCost_Disp = pyo.Param(m.Dispatchable, initialize=investment_cost_disp_init)

    def investment_cost_nd_init(m, t):
        return tech_params.loc[t, 'InvestmentCost']
    m.InvestmentCost_ND = pyo.Param(m.NonDispatchable, initialize=investment_cost_nd_init)

    def marginal_cost_disp_init(m, t):
        return tech_params.loc[t, 'MarginalCost']
    m.MarginalCost_Disp = pyo.Param(m.Dispatchable, initialize=marginal_cost_disp_init)

    def fixed_cost_disp_init(m, t):
        return tech_params.loc[t, 'FixedCost']
    m.FixedCost_Disp = pyo.Param(m.Dispatchable, initialize=fixed_cost_disp_init)

    def fixed_cost_nd_init(m, t):
        return tech_params.loc[t, 'FixedCost']
    m.FixedCost_ND = pyo.Param(m.NonDispatchable, initialize=fixed_cost_nd_init)

    def max_install_nd_init(m, t):
        return tech_params.loc[t, 'MaxInstallable']
    m.MaxInstallable_ND = pyo.Param(m.NonDispatchable, initialize=max_install_nd_init)

    # Storage params
    def inv_cost_energy_init(m, s):
        return sto_params.loc[s, 'InvestmentCostEnergy']
    m.InvestmentCostEnergy = pyo.Param(m.Storages, initialize=inv_cost_energy_init)

    def inv_cost_power_init(m, s):
        return sto_params.loc[s, 'InvestmentCostPower']
    m.InvestmentCostPower = pyo.Param(m.Storages, initialize=inv_cost_power_init)

    def marginal_cost_sto_init(m, s):
        return sto_params.loc[s, 'MarginalCost']
    m.MarginalCost_Sto = pyo.Param(m.Storages, initialize=marginal_cost_sto_init)

    def fixed_cost_sto_init(m, s):
        return sto_params.loc[s, 'FixedCost']
    m.FixedCost_Sto = pyo.Param(m.Storages, initialize=fixed_cost_sto_init)

    def efficiency_sto_init(m, s):
        return sto_params.loc[s, 'Efficiency']
    m.Efficiency_Sto = pyo.Param(m.Storages, initialize=efficiency_sto_init)

    def max_energy_sto_init(m, s):
        return sto_params.loc[s, 'MaxEnergy']
    m.MaxEnergy_Sto = pyo.Param(m.Storages, initialize=max_energy_sto_init)

    def max_power_sto_init(m, s):
        return sto_params.loc[s, 'MaxPower']
    m.MaxPower_Sto = pyo.Param(m.Storages, initialize=max_power_sto_init)

    # Time series params
    load_data = data['load']
    def load_init(m, h):
        return float(load_data[h - 1])
    m.Load = pyo.Param(m.Hours, initialize=load_init)

    # Availability (2D: NonDispatchable x Hours)
    avail_df = data['availability']
    avail_dict = {(row['Technologies'], row['Hours']): row['Availability']
                  for _, row in avail_df.iterrows()}
    def availability_init(m, t, h):
        return avail_dict.get((t, h), 0.0)
    m.Availability = pyo.Param(m.NonDispatchable, m.Hours, initialize=availability_init)

    # Correction factor
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
        op_cost_disp = sum(m.MarginalCost_Disp[t] * m.G_D[t, h] 
                          for t in m.Dispatchable for h in m.Hours)
        op_cost_sto = sum(m.MarginalCost_Sto[s] * (m.STO_IN[s, h] + m.STO_OUT[s, h])
                         for s in m.Storages for h in m.Hours)
        cu_cost = CURTAILMENT_COST * sum(m.CU[t, h] 
                                         for t in m.NonDispatchable for h in m.Hours)
        inf_cost = INFEAS_COST * sum(m.G_INF[h] for h in m.Hours)

        inv_cost_disp = cf * sum(m.InvestmentCost_Disp[t] * m.N_D[t] for t in m.Dispatchable)
        inv_cost_nd = cf * sum(m.InvestmentCost_ND[t] * m.N_ND[t] for t in m.NonDispatchable)
        inv_cost_sto_e = cf * sum(m.InvestmentCostEnergy[s] * m.N_STO_E[s] for s in m.Storages)
        inv_cost_sto_p = cf * sum(m.InvestmentCostPower[s] * m.N_STO_P[s] for s in m.Storages)

        fixed_disp = cf * sum(m.FixedCost_Disp[t] * m.N_D[t] for t in m.Dispatchable)
        fixed_nd = cf * sum(m.FixedCost_ND[t] * m.N_ND[t] for t in m.NonDispatchable)
        fixed_sto = cf * sum(m.FixedCost_Sto[s] * 0.5 * (m.N_STO_P[s] + m.N_STO_E[s]) 
                            for s in m.Storages)

        return (op_cost_disp + op_cost_sto + cu_cost + inf_cost +
                inv_cost_disp + inv_cost_nd + inv_cost_sto_e + inv_cost_sto_p +
                fixed_disp + fixed_nd + fixed_sto)
    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    def energy_balance_rule(m, h):
        gen_disp = sum(m.G_D[t, h] for t in m.Dispatchable)
        gen_nd = sum(m.G_ND[t, h] for t in m.NonDispatchable)
        sto_out = sum(m.STO_OUT[s, h] for s in m.Storages)
        sto_in = sum(m.STO_IN[s, h] for s in m.Storages)
        return gen_disp + gen_nd + sto_out + m.G_INF[h] == sto_in + m.Load[h]
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

    # Storage constraints
    def max_withdraw_sto_rule(m, s, h):
        return m.STO_IN[s, h] <= m.N_STO_P[s]
    m.MaxWithdrawSto = pyo.Constraint(m.Storages, m.Hours, rule=max_withdraw_sto_rule)

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

    # Storage balance with cyclic lag
    hours_list = list(m.Hours)
    def storage_balance_rule(m, s, h):
        sqrt_eff = pyo.sqrt(m.Efficiency_Sto[s])
        h_prev = hours_list[-1] if h == hours_list[0] else hours_list[hours_list.index(h) - 1]
        return m.STO_L[s, h] == (m.STO_L[s, h_prev] + 
                                 sqrt_eff * m.STO_IN[s, h] - 
                                 m.STO_OUT[s, h] / sqrt_eff)
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
    solver = SolverFactory('appsi_highs')
    t_solver_init = time.time() - t0
    print(f"  Solver init: {t_solver_init:.2f}s")

    t0 = time.time()
    result = solver.solve(model, tee=False)
    t_solve = time.time() - t0
    print(f"  Solve time: {t_solve:.2f}s")

    print(f"\nStatus: {result.solver.termination_condition}")
    print(f"Objective: €{pyo.value(model.objective):,.0f}")


if __name__ == '__main__':
    main()
