#!/usr/bin/env python3
"""
Convert DIETER.jl test data to nimopt long-format CSVs.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Input/output paths
INPUT_DIR = Path("/home/carlos/projects/Dieter.jl/testdata/base")
OUTPUT_DIR = Path("/home/carlos/projects/nimopt/examples/dieter/data")

# Interest rate for annuity calculation
INTEREST_RATE = 0.07

def annuity(i: float, lifetime: float) -> float:
    """Calculate annuity factor."""
    if lifetime <= 0:
        return 0
    return i * ((1 + i) ** lifetime) / (((1 + i) ** lifetime) - 1)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # =========================================================================
    # Technologies
    # =========================================================================
    tech_df = pd.read_csv(INPUT_DIR / "technologies.csv")
    
    # Save technologies with attributes
    tech_df.to_csv(OUTPUT_DIR / "technologies.csv", index=False)
    print(f"Technologies: {len(tech_df)} rows")
    
    # Calculate investment costs and marginal costs
    tech_params = []
    for _, row in tech_df.iterrows():
        t = row['Technologies']
        oc = row['OvernightCost']
        lt = row['Lifetime']
        fc = row['FuelCost']
        eff = row['Efficiency']
        cc = row['CarbonContent']
        co2 = row['CO2_price']
        vc = row['VariableCost']
        
        # Investment cost = overnight cost * annuity factor
        inv_cost = oc * annuity(INTEREST_RATE, lt)
        
        # Marginal cost = fuel/efficiency + (carbon * CO2 price)/efficiency + variable cost
        mc = fc / eff + (cc * co2) / eff + vc
        
        tech_params.append({
            'Technologies': t,
            'InvestmentCost': inv_cost,
            'MarginalCost': mc,
            'Efficiency': eff,
            'FixedCost': row['FixedCost'],
            'MaxInstallable': row['MaxInstallable'] if row['MaxInstallable'] >= 0 else 1e9,
            'MaxEnergy': row['MaxEnergy'] if row['MaxEnergy'] >= 0 else 1e15,
        })
    
    tech_params_df = pd.DataFrame(tech_params)
    tech_params_df.to_csv(OUTPUT_DIR / "tech_params.csv", index=False)
    
    # =========================================================================
    # Storages
    # =========================================================================
    sto_df = pd.read_csv(INPUT_DIR / "storages.csv")
    sto_params = []
    for _, row in sto_df.iterrows():
        s = row['Storages']
        lt = row['Lifetime']
        
        # Investment costs for energy and power capacity
        inv_energy = row['OvernightCostEnergy'] * annuity(INTEREST_RATE, lt)
        inv_power = row['OvernightCostPower'] * annuity(INTEREST_RATE, lt)
        
        sto_params.append({
            'Storages': s,
            'InvestmentCostEnergy': inv_energy,
            'InvestmentCostPower': inv_power,
            'MarginalCost': row['MarginalCost'],
            'Efficiency': row['Efficiency'],
            'FixedCost': row['FixedCost'],
            'MaxEnergy': row['MaxEnergy'] if row['MaxEnergy'] >= 0 else 1e12,
            'MaxPower': row['MaxPower'] if row['MaxPower'] >= 0 else 1e12,
        })
    
    sto_params_df = pd.DataFrame(sto_params)
    sto_params_df.to_csv(OUTPUT_DIR / "sto_params.csv", index=False)
    print(f"Storages: {len(sto_df)} rows")
    
    # =========================================================================
    # Load (hourly demand)
    # =========================================================================
    load_df = pd.read_csv(INPUT_DIR / "load.csv")
    # Add hour index (1-based)
    load_df['Hours'] = range(1, len(load_df) + 1)
    load_df.to_csv(OUTPUT_DIR / "load.csv", index=False)
    print(f"Load: {len(load_df)} hours")
    
    # =========================================================================
    # Availability (capacity factors for renewables)
    # =========================================================================
    avail_df = pd.read_csv(INPUT_DIR / "availability.csv")
    # Melt to long format
    avail_long = []
    for h, row in enumerate(avail_df.itertuples(index=False), start=1):
        for col in avail_df.columns:
            avail_long.append({
                'Technologies': col,
                'Hours': h,
                'Availability': getattr(row, col.replace('-', '_') if '-' in col else col)
            })
    
    avail_long_df = pd.DataFrame(avail_long)
    avail_long_df.to_csv(OUTPUT_DIR / "availability.csv", index=False)
    print(f"Availability: {len(avail_long_df)} rows (tech x hours)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\nOutput files:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
