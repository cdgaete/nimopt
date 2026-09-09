# The network a fidelity check restates

`elec_s_10.npz` holds a PyPSA capacity expansion network as plain arrays, and
`elec_s_10_reference.json` holds what PyPSA's own model carries over it: rows,
columns and nonzeros overall and per constraint family, and the objective the
network solves to.

The two files exist so the restatement is checked without PyPSA installed and
without the network file present. `nimopt` never imports PyPSA.

Source: `elec_s_10_ec_lcopt_Co2L-4H.nc` — 30 buses, 36 snapshots, 41
generators, 10 lines, 40 links, 20 stores, 2 storage units. PyPSA's model over
it is 10,499 rows by 5,105 columns with 21,394 nonzeros across 26 constraint
families, and solves to 2,662,944,681.859 beside a constant of
368,025,607.863. The JSON records which PyPSA wrote it.

Regenerate both with the network's path:

```bash
python benchmarks/pypsa_reference.py <path to elec_s_10_ec_lcopt_Co2L-4H.nc>
```

A regenerated reference whose numbers differ is a different model, and every
check that reads it moves with it.
