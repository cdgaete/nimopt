#!/usr/bin/env python3
"""
Benchmark nimopt solver paths: Direct vs LP file.

Compares:
1. HiGHS Direct solver (matrices loaded directly via highspy)
2. HiGHS via LP file (Python LP writer)
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "examples" / "dieter" / "data"
CURTAILMENT_COST = 0.1
INFEAS_COST = 10000


def load_data(n_hours: int):
    """Load data for benchmark."""
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


def build_model(data, corr_factor):
    """Build DIETER model and return it."""
    import nimopt as no
    from nimopt import sqrt, Sum

    disp = data['dispatchable']
    nd = data['non_dispatchable']
    sto = data['storages']
    hours = data['hours']
    tech_params = data['tech_params']
    sto_params = data['sto_params']

    Dispatchable = no.Set('Dispatchable', disp)
    NonDispatchable = no.Set('NonDispatchable', nd)
    Storages = no.Set('Storages', sto)
    Hours = no.Set('Hours', hours)

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

    m.eq('EnergyBalance', 
         Sum(Dispatchable, G_D[Dispatchable, Hours]) + 
         Sum(NonDispatchable, G_ND[NonDispatchable, Hours]) + 
         Sum(Storages, STO_OUT[Storages, Hours]) + G_INF[Hours] == 
         Sum(Storages, STO_IN[Storages, Hours]) + Load[Hours])
    m.eq('MaxGenDisp', G_D[Dispatchable, Hours] <= N_D[Dispatchable])
    m.eq('MaxGenNonDisp', 
         G_ND[NonDispatchable, Hours] + CU[NonDispatchable, Hours] == 
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

    return m


def benchmark_direct(model):
    """Benchmark HiGHS Direct solver."""
    from nimopt.solvers import HiGHSDirectSolver
    
    t0 = time.time()
    solver = HiGHSDirectSolver(use_rust=True)
    solver.load_model(model)
    t_load = time.time() - t0
    
    t0 = time.time()
    result = solver.solve()
    t_solve = time.time() - t0
    
    return {'load': t_load, 'solve': t_solve, 'objective': result.objective_value}


def benchmark_lp_python(model):
    """Benchmark HiGHS via LP file (Python writer)."""
    from nimopt.solvers import HiGHSSolver
    
    lp_file = tempfile.NamedTemporaryFile(suffix='.lp', delete=False).name
    
    t0 = time.time()
    model.to_lp(lp_file, use_rust=False)  # Python LP writer
    t_write = time.time() - t0
    
    t0 = time.time()
    solver = HiGHSSolver()
    solver.read_lp(lp_file)
    solver.set_model(model)
    t_read = time.time() - t0
    
    t0 = time.time()
    result = solver.solve()
    t_solve = time.time() - t0
    
    return {'write': t_write, 'read': t_read, 'solve': t_solve, 
            'objective': result.objective_value}


def main():
    parser = argparse.ArgumentParser(description='Benchmark nimopt solver paths')
    parser.add_argument('--hours', type=int, nargs='+', default=[168, 720, 2000, 4000, 8760])
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("nimopt Solver Benchmark: Direct vs LP File")
    print("=" * 70)

    results = []
    for n_hours in args.hours:
        corr_factor = n_hours / 8760
        print(f"\n--- {n_hours} hours ({n_hours/24:.1f} days) ---")

        # Load data and build model
        t0 = time.time()
        data = load_data(n_hours)
        t_data = time.time() - t0
        
        t0 = time.time()
        model = build_model(data, corr_factor)
        t_build = time.time() - t0
        
        print(f"Data load: {t_data:.3f}s, Model build: {t_build:.3f}s")

        # Direct solver
        print("\nDirect solver (matrices -> HiGHS):")
        r_direct = benchmark_direct(model)
        print(f"  Load:  {r_direct['load']:.3f}s")
        print(f"  Solve: {r_direct['solve']:.3f}s")
        print(f"  Total: {r_direct['load'] + r_direct['solve']:.3f}s")
        print(f"  Obj:   €{r_direct['objective']:,.0f}")

        # LP file solver
        print("\nLP file solver (Python writer -> HiGHS):")
        r_lp = benchmark_lp_python(model)
        print(f"  Write: {r_lp['write']:.3f}s")
        print(f"  Read:  {r_lp['read']:.3f}s")
        print(f"  Solve: {r_lp['solve']:.3f}s")
        print(f"  Total: {r_lp['write'] + r_lp['read'] + r_lp['solve']:.3f}s")
        print(f"  Obj:   €{r_lp['objective']:,.0f}")

        results.append({
            'hours': n_hours,
            'build': t_build,
            'direct_load': r_direct['load'],
            'direct_solve': r_direct['solve'],
            'direct_total': r_direct['load'] + r_direct['solve'],
            'lp_write': r_lp['write'],
            'lp_read': r_lp['read'],
            'lp_solve': r_lp['solve'],
            'lp_total': r_lp['write'] + r_lp['read'] + r_lp['solve'],
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    print("\nModel Build Time (lazy constraint storage):")
    print(df[['hours', 'build']].to_string(index=False))
    
    print("\nDirect Solver (load matrices + solve):")
    print(df[['hours', 'direct_load', 'direct_solve', 'direct_total']].to_string(index=False))
    
    print("\nLP File Solver (write LP + read LP + solve):")
    print(df[['hours', 'lp_write', 'lp_read', 'lp_solve', 'lp_total']].to_string(index=False))
    
    print("\nSpeedup (LP total / Direct total):")
    df['speedup'] = df['lp_total'] / df['direct_total']
    print(df[['hours', 'speedup']].to_string(index=False))


if __name__ == '__main__':
    main()
