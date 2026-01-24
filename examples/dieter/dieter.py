#!/usr/bin/env python3
"""
DIETER Energy System Model - nimopt Implementation

Dispatch and Investment Evaluation Tool with Endogenous Renewables.
Based on DIETER.jl from DIW Berlin.

This implementation demonstrates nimopt's capabilities for energy modeling:
- Technology subsets (Renewables, Dispatchable, etc.)
- Time-coupled constraints with lag() for storage balance
- Parameter math functions (sqrt for storage efficiency)
- Large-scale optimization with HiGHS

Usage:
    python dieter.py                    # Full year (8760 hours)
    python dieter.py --hours 168        # One week
    python dieter.py --hours 24         # One day
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import nimopt as no
from nimopt import sqrt, Sum
from nimopt.solvers import HiGHSDirectSolver

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
CORR_FACTOR = 1.0  # Correction factor for subsampling
CURTAILMENT_COST = 0.1  # €/MWh curtailment penalty
INFEAS_COST = 10000  # €/MWh for unserved energy (slack)


# =============================================================================
# Data Loading
# =============================================================================

def load_sets(n_hours: int = 8760):
    """Load all sets from CSV files."""
    
    # Technologies - full set and subsets
    tech_df = pd.read_csv(DATA_DIR / "technologies.csv")
    
    Tech = no.Set('Tech', tech_df['Technologies'].tolist())
    
    Renewables = no.Set.from_csv(
        'Renewables', 
        str(DATA_DIR / "technologies.csv"),
        'Technologies',
        filter_col='Renewable',
        filter_val=1
    )
    
    Conventional = no.Set.from_csv(
        'Conventional',
        str(DATA_DIR / "technologies.csv"),
        'Technologies', 
        filter_col='Renewable',
        filter_val=0
    )
    
    Dispatchable = no.Set.from_csv(
        'Dispatchable',
        str(DATA_DIR / "technologies.csv"),
        'Technologies',
        filter_col='Dispatchable',
        filter_val=1
    )
    
    NonDispatchable = no.Set.from_csv(
        'NonDispatchable',
        str(DATA_DIR / "technologies.csv"),
        'Technologies',
        filter_col='Dispatchable',
        filter_val=0
    )
    
    # Storages
    sto_df = pd.read_csv(DATA_DIR / "sto_params.csv")
    Storages = no.Set('Storages', sto_df['Storages'].tolist())
    
    # Hours (with optional subsampling)
    Hours = no.Set('Hours', list(range(1, n_hours + 1)))
    
    return {
        'Tech': Tech,
        'Renewables': Renewables,
        'Conventional': Conventional,
        'Dispatchable': Dispatchable,
        'NonDispatchable': NonDispatchable,
        'Storages': Storages,
        'Hours': Hours,
    }


def load_params(sets: dict):
    """Load all parameters from CSV files."""
    Tech = sets['Tech']
    Dispatchable = sets['Dispatchable']
    NonDispatchable = sets['NonDispatchable']
    Storages = sets['Storages']
    Hours = sets['Hours']
    
    n_hours = len(Hours)
    
    # Technology parameters
    tech_df = pd.read_csv(DATA_DIR / "tech_params.csv")
    tech_df = tech_df.set_index('Technologies')
    
    # Create params for all technologies
    InvestmentCost = no.Param('InvestmentCost', [Tech], 
                              [tech_df.loc[t, 'InvestmentCost'] for t in Tech])
    MarginalCost_Tech = no.Param('MarginalCost_Tech', [Tech],
                                  [tech_df.loc[t, 'MarginalCost'] for t in Tech])
    FixedCost = no.Param('FixedCost', [Tech],
                         [tech_df.loc[t, 'FixedCost'] for t in Tech])
    MaxInstallable = no.Param('MaxInstallable', [Tech],
                              [tech_df.loc[t, 'MaxInstallable'] for t in Tech])
    
    # Storage parameters  
    sto_df = pd.read_csv(DATA_DIR / "sto_params.csv")
    sto_df = sto_df.set_index('Storages')
    
    InvestmentCostEnergy = no.Param('InvestmentCostEnergy', [Storages],
                                    [sto_df.loc[s, 'InvestmentCostEnergy'] for s in Storages])
    InvestmentCostPower = no.Param('InvestmentCostPower', [Storages],
                                   [sto_df.loc[s, 'InvestmentCostPower'] for s in Storages])
    MarginalCost_Sto = no.Param('MarginalCost_Sto', [Storages],
                                [sto_df.loc[s, 'MarginalCost'] for s in Storages])
    FixedCost_Sto = no.Param('FixedCost_Sto', [Storages],
                             [sto_df.loc[s, 'FixedCost'] for s in Storages])
    Efficiency_Sto = no.Param('Efficiency_Sto', [Storages],
                              [sto_df.loc[s, 'Efficiency'] for s in Storages])
    MaxEnergy_Sto = no.Param('MaxEnergy_Sto', [Storages],
                             [sto_df.loc[s, 'MaxEnergy'] for s in Storages])
    MaxPower_Sto = no.Param('MaxPower_Sto', [Storages],
                            [sto_df.loc[s, 'MaxPower'] for s in Storages])
    
    # Load (hourly demand)
    load_df = pd.read_csv(DATA_DIR / "load.csv")
    Load = no.Param('Load', [Hours],
                    load_df.loc[:n_hours-1, 'Load'].values)
    
    # Availability (capacity factors) - 2D: NonDispatchable x Hours
    avail_df = pd.read_csv(DATA_DIR / "availability.csv")
    
    # Filter to only NonDispatchable technologies and requested hours
    nd_list = list(NonDispatchable.elements)
    avail_filtered = avail_df[
        (avail_df['Technologies'].isin(nd_list)) & 
        (avail_df['Hours'] <= n_hours)
    ]
    
    # Build 2D array in correct order
    avail_matrix = np.zeros((len(NonDispatchable), n_hours))
    for i, t in enumerate(NonDispatchable):
        tech_data = avail_filtered[avail_filtered['Technologies'] == t]
        tech_data = tech_data.sort_values('Hours')
        avail_matrix[i, :] = tech_data['Availability'].values
    
    Availability = no.Param('Availability', [NonDispatchable, Hours], avail_matrix)
    
    # Marginal costs for dispatchable technologies only
    MarginalCost_Disp = no.Param('MarginalCost_Disp', [Dispatchable],
                                  [tech_df.loc[t, 'MarginalCost'] for t in Dispatchable])
    
    # Investment/Fixed costs for subsets
    InvestmentCost_Disp = no.Param('InvestmentCost_Disp', [Dispatchable],
                                    [tech_df.loc[t, 'InvestmentCost'] for t in Dispatchable])
    InvestmentCost_ND = no.Param('InvestmentCost_ND', [NonDispatchable],
                                  [tech_df.loc[t, 'InvestmentCost'] for t in NonDispatchable])
    FixedCost_Disp = no.Param('FixedCost_Disp', [Dispatchable],
                              [tech_df.loc[t, 'FixedCost'] for t in Dispatchable])
    FixedCost_ND = no.Param('FixedCost_ND', [NonDispatchable],
                            [tech_df.loc[t, 'FixedCost'] for t in NonDispatchable])
    MaxInstallable_ND = no.Param('MaxInstallable_ND', [NonDispatchable],
                                  [tech_df.loc[t, 'MaxInstallable'] for t in NonDispatchable])
    
    return {
        # Technology params
        'InvestmentCost': InvestmentCost,
        'MarginalCost_Tech': MarginalCost_Tech,
        'FixedCost': FixedCost,
        'MaxInstallable': MaxInstallable,
        
        # Subset-specific params
        'InvestmentCost_Disp': InvestmentCost_Disp,
        'InvestmentCost_ND': InvestmentCost_ND,
        'FixedCost_Disp': FixedCost_Disp,
        'FixedCost_ND': FixedCost_ND,
        'MarginalCost_Disp': MarginalCost_Disp,
        'MaxInstallable_ND': MaxInstallable_ND,
        
        # Storage params
        'InvestmentCostEnergy': InvestmentCostEnergy,
        'InvestmentCostPower': InvestmentCostPower,
        'MarginalCost_Sto': MarginalCost_Sto,
        'FixedCost_Sto': FixedCost_Sto,
        'Efficiency_Sto': Efficiency_Sto,
        'MaxEnergy_Sto': MaxEnergy_Sto,
        'MaxPower_Sto': MaxPower_Sto,
        
        # Time series
        'Load': Load,
        'Availability': Availability,
    }


# =============================================================================
# Model Building
# =============================================================================

def build_model(sets: dict, params: dict, corr_factor: float = 1.0):
    """
    Build the DIETER optimization model.
    
    Parameters
    ----------
    sets : dict
        Dictionary of Set objects.
    params : dict
        Dictionary of Param objects.
    corr_factor : float
        Correction factor for investment costs when using subsampling.
        
    Returns
    -------
    no.Model
        The optimization model ready for solving.
    """
    
    # Unpack sets
    Dispatchable = sets['Dispatchable']
    NonDispatchable = sets['NonDispatchable']
    Storages = sets['Storages']
    Hours = sets['Hours']
    
    # Unpack params
    InvestmentCost_Disp = params['InvestmentCost_Disp']
    InvestmentCost_ND = params['InvestmentCost_ND']
    InvestmentCostEnergy = params['InvestmentCostEnergy']
    InvestmentCostPower = params['InvestmentCostPower']
    MarginalCost_Disp = params['MarginalCost_Disp']
    MarginalCost_Sto = params['MarginalCost_Sto']
    FixedCost_Disp = params['FixedCost_Disp']
    FixedCost_ND = params['FixedCost_ND']
    FixedCost_Sto = params['FixedCost_Sto']
    Efficiency_Sto = params['Efficiency_Sto']
    MaxInstallable_ND = params['MaxInstallable_ND']
    MaxEnergy_Sto = params['MaxEnergy_Sto']
    MaxPower_Sto = params['MaxPower_Sto']
    Load = params['Load']
    Availability = params['Availability']
    
    # Compute sqrt(efficiency) for storage balance equations
    sqrt_eff = sqrt(Efficiency_Sto)
    
    # =========================================================================
    # Create Model
    # =========================================================================
    
    m = no.Model(sense='minimize')
    
    # =========================================================================
    # Variables
    # =========================================================================
    
    # Generation variables (separate for dispatchable and non-dispatchable)
    G_D = m.var('G_D', [Dispatchable, Hours], lb=0)    # Dispatchable generation
    G_ND = m.var('G_ND', [NonDispatchable, Hours], lb=0)  # Non-dispatchable generation
    
    # Curtailment (only for non-dispatchable)
    CU = m.var('CU', [NonDispatchable, Hours], lb=0)
    
    # Storage variables
    STO_IN = m.var('STO_IN', [Storages, Hours], lb=0)   # Storage charging
    STO_OUT = m.var('STO_OUT', [Storages, Hours], lb=0)  # Storage discharging
    STO_L = m.var('STO_L', [Storages, Hours], lb=0)      # Storage level
    
    # Capacity variables
    N_D = m.var('N_D', [Dispatchable], lb=0)         # Dispatchable capacity
    N_ND = m.var('N_ND', [NonDispatchable], lb=0)    # Non-dispatchable capacity
    N_STO_E = m.var('N_STO_E', [Storages], lb=0)     # Storage energy capacity
    N_STO_P = m.var('N_STO_P', [Storages], lb=0)     # Storage power capacity
    
    # Slack variable for infeasibility
    G_INF = m.var('G_INF', [Hours], lb=0)
    
    # =========================================================================
    # Objective Function
    # =========================================================================
    # Minimize: operating costs + investment costs + fixed costs
    
    # Operating costs (generation)
    op_cost_disp = Sum(Dispatchable, Hours, MarginalCost_Disp[Dispatchable] * G_D[Dispatchable, Hours])
    op_cost_sto = Sum(Storages, Hours, MarginalCost_Sto[Storages] * (STO_IN[Storages, Hours] + STO_OUT[Storages, Hours]))
    
    # Curtailment cost
    cu_cost = CURTAILMENT_COST * Sum(NonDispatchable, Hours, CU[NonDispatchable, Hours])
    
    # Infeasibility cost
    inf_cost = INFEAS_COST * Sum(Hours, G_INF[Hours])
    
    # Investment costs (annualized, scaled by correction factor)
    inv_cost_disp = corr_factor * Sum(Dispatchable, InvestmentCost_Disp[Dispatchable] * N_D[Dispatchable])
    inv_cost_nd = corr_factor * Sum(NonDispatchable, InvestmentCost_ND[NonDispatchable] * N_ND[NonDispatchable])
    inv_cost_sto_e = corr_factor * Sum(Storages, InvestmentCostEnergy[Storages] * N_STO_E[Storages])
    inv_cost_sto_p = corr_factor * Sum(Storages, InvestmentCostPower[Storages] * N_STO_P[Storages])
    
    # Fixed costs (scaled by correction factor)
    fixed_cost_disp = corr_factor * Sum(Dispatchable, FixedCost_Disp[Dispatchable] * N_D[Dispatchable])
    fixed_cost_nd = corr_factor * Sum(NonDispatchable, FixedCost_ND[NonDispatchable] * N_ND[NonDispatchable])
    fixed_cost_sto = corr_factor * Sum(Storages, FixedCost_Sto[Storages] * 0.5 * (N_STO_P[Storages] + N_STO_E[Storages]))
    
    # Total objective
    m.objective = (op_cost_disp + op_cost_sto + cu_cost + inf_cost +
                   inv_cost_disp + inv_cost_nd + inv_cost_sto_e + inv_cost_sto_p +
                   fixed_cost_disp + fixed_cost_nd + fixed_cost_sto)
    
    # =========================================================================
    # Constraints
    # =========================================================================
    
    # Energy Balance: generation = load + storage charging - storage discharging
    # Note: Put Sum (returns LinearExpr) first to ensure proper operator handling
    m.eq('EnergyBalance',
         Sum(Dispatchable, G_D[Dispatchable, Hours]) +
         Sum(NonDispatchable, G_ND[NonDispatchable, Hours]) +
         Sum(Storages, STO_OUT[Storages, Hours]) +
         G_INF[Hours]
         ==
         Sum(Storages, STO_IN[Storages, Hours]) + Load[Hours]
    )
    
    # Dispatchable generation limit: G_D <= N_D
    m.eq('MaxGenDisp', G_D[Dispatchable, Hours] <= N_D[Dispatchable])
    
    # Non-dispatchable generation: G_ND + CU == Availability * N_ND
    m.eq('MaxGenNonDisp', 
         G_ND[NonDispatchable, Hours] + CU[NonDispatchable, Hours] 
         == Availability[NonDispatchable, Hours] * N_ND[NonDispatchable])
    
    # Maximum installable capacity for non-dispatchable
    m.eq('MaxInstallND', N_ND[NonDispatchable] <= MaxInstallable_ND[NonDispatchable])
    
    # Storage constraints
    m.eq('MaxWithdrawSto', STO_IN[Storages, Hours] <= N_STO_P[Storages])
    m.eq('MaxGenSto', STO_OUT[Storages, Hours] <= N_STO_P[Storages])
    m.eq('MaxLevelSto', STO_L[Storages, Hours] <= N_STO_E[Storages])
    m.eq('MaxEnergySto', N_STO_E[Storages] <= MaxEnergy_Sto[Storages])
    m.eq('MaxPowerSto', N_STO_P[Storages] <= MaxPower_Sto[Storages])
    
    # Storage Balance (with lag for temporal coupling)
    # STO_L[s,h] == STO_L[s,h-1] + sqrt(eff)*STO_IN[s,h] - STO_OUT[s,h]/sqrt(eff)
    m.eq('StorageBalance',
         STO_L[Storages, Hours] == 
         STO_L[Storages, Hours.lag(1, cyclic=True)] +
         sqrt_eff[Storages] * STO_IN[Storages, Hours] -
         STO_OUT[Storages, Hours] / sqrt_eff[Storages]
    )
    
    return m


# =============================================================================
# Solution Analysis
# =============================================================================

def analyze_solution(solver, sets: dict, params: dict):
    """Analyze and print solution summary."""
    
    Dispatchable = sets['Dispatchable']
    NonDispatchable = sets['NonDispatchable']
    Storages = sets['Storages']
    Hours = sets['Hours']
    
    sol = solver.get_solution()
    
    def get_val(arr):
        """Get float value from nimblend Array or scalar."""
        if hasattr(arr, 'values'):
            return float(arr.values)
        return float(arr)
    
    print("\n" + "=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)
    
    # Installed capacities - Dispatchable
    print("\nInstalled Capacities (MW):")
    print("-" * 40)
    N_D_vals = sol.var('N_D')
    for t in Dispatchable:
        val = get_val(N_D_vals.sel({Dispatchable.name: t}))
        if val > 0.1:
            print(f"  {t:20s}: {val:12.1f}")
    
    # Installed capacities - Non-dispatchable
    N_ND_vals = sol.var('N_ND')
    for t in NonDispatchable:
        val = get_val(N_ND_vals.sel({NonDispatchable.name: t}))
        if val > 0.1:
            print(f"  {t:20s}: {val:12.1f}")
    
    # Storage capacities
    print("\nStorage Capacities:")
    print("-" * 40)
    N_STO_E_vals = sol.var('N_STO_E')
    N_STO_P_vals = sol.var('N_STO_P')
    for s in Storages:
        e_val = get_val(N_STO_E_vals.sel({Storages.name: s}))
        p_val = get_val(N_STO_P_vals.sel({Storages.name: s}))
        print(f"  {s:20s}: {e_val:12.1f} MWh, {p_val:10.1f} MW")
    
    # Total generation by technology
    print("\nTotal Generation (TWh):")
    print("-" * 40)
    
    G_D_vals = sol.var('G_D')
    total_disp = 0
    for t in Dispatchable:
        total = get_val(G_D_vals.sel({Dispatchable.name: t}).sum()) / 1e6  # MWh to TWh
        if total > 0.001:
            print(f"  {t:20s}: {total:12.3f}")
        total_disp += total
    
    G_ND_vals = sol.var('G_ND')
    total_nd = 0
    for t in NonDispatchable:
        total = get_val(G_ND_vals.sel({NonDispatchable.name: t}).sum()) / 1e6
        if total > 0.001:
            print(f"  {t:20s}: {total:12.3f}")
        total_nd += total
    
    print(f"\n  Total Dispatchable:    {total_disp:12.3f} TWh")
    print(f"  Total Non-Dispatchable: {total_nd:12.3f} TWh")
    
    # Curtailment
    CU_vals = sol.var('CU')
    total_cu = get_val(CU_vals.sum())
    print(f"\n  Total Curtailment:      {total_cu / 1e6:12.3f} TWh")
    
    # Infeasibility
    G_INF_vals = sol.var('G_INF')
    total_inf = get_val(G_INF_vals.sum())
    print(f"\n  Total Infeasibility:    {total_inf:12.1f} MWh")
    if total_inf > 0.1:
        print(f"  WARNING: Unserved energy present")
    
    # Total load
    Load = params['Load']
    total_load = float(Load.values.sum()) / 1e6
    print(f"\n  Total Load:             {total_load:12.3f} TWh")
    
    return sol


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='DIETER Energy System Model')
    parser.add_argument('--hours', type=int, default=168,
                        help='Number of hours to model (default: 168 = 1 week)')
    parser.add_argument('--rust', action='store_true', default=True,
                        help='Use Rust acceleration (default: True)')
    parser.add_argument('--no-rust', action='store_false', dest='rust',
                        help='Disable Rust acceleration')
    args = parser.parse_args()
    
    n_hours = args.hours
    use_rust = args.rust
    
    # Correction factor for investment costs when using subsampling
    corr_factor = n_hours / 8760
    
    print(f"\n{'='*60}")
    print(f"DIETER Energy System Model - nimopt Implementation")
    print(f"{'='*60}")
    print(f"Hours: {n_hours} ({n_hours/24:.1f} days)")
    print(f"Correction factor: {corr_factor:.4f}")
    print(f"Rust acceleration: {use_rust}")
    
    # Load data
    print("\nLoading data...")
    t0 = time.time()
    sets = load_sets(n_hours)
    params = load_params(sets)
    t_load = time.time() - t0
    print(f"  Data loading: {t_load:.2f}s")
    
    # Build model
    print("\nBuilding model...")
    t0 = time.time()
    model = build_model(sets, params, corr_factor)
    t_build = time.time() - t0
    print(f"  Model building: {t_build:.2f}s")
    
    # Solve
    print("\nSolving...")
    t0 = time.time()
    solver = HiGHSDirectSolver(use_rust=use_rust)
    solver.load_model(model)
    t_load_solver = time.time() - t0
    print(f"  Solver loading: {t_load_solver:.2f}s")
    
    t0 = time.time()
    result = solver.solve()
    t_solve = time.time() - t0
    print(f"  Optimization: {t_solve:.2f}s")
    
    print(f"\nStatus: {result.status.name}")
    print(f"Objective: €{result.objective_value:,.0f}")
    
    if result.status.name == 'OPTIMAL':
        analyze_solution(solver, sets, params)
    
    return result


if __name__ == '__main__':
    main()
