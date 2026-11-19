---
title: Interior point and first-order methods
description: The HiGHS simplex, IPX, HiPO and PDLP methods through the options of nimopt, the memory each requires, and how to install a HiGHS with HiPO and GPU support.
---

# Interior point and first-order methods

`method=` selects the algorithm HiGHS runs on the assembled matrix. Four
values select one algorithm each, and three further options configure the
interior point and first-order methods. Every setting on this page runs in
the test suite against the bundled models, and each solve returns the optimum
of the model. Which method is fastest or smallest depends on the model.

| Option | Choices | Applies to |
| --- | --- | --- |
| `method` | `choose`, `simplex`, `barrier`, `hipo`, `pdlp` | every solve |
| `newton_system` | `choose`, `augmented`, `normaleq` | `hipo` |
| `crossover` | `choose`, `off`, `on` | `barrier` and `hipo` |
| `pdlp_tol` | a relative tolerance | `pdlp` |

`barrier` runs IPX, the HiGHS interior point method built on a
preconditioned conjugate gradient. `hipo` runs HiPO, an interior point method
built on a direct factorization of the Newton system, parallel across the
elimination tree. `newton_system` selects the augmented system or the normal
equations, and `choose` leaves the selection to the solver. HiPO stores that
factorization in memory and requires more memory than IPX on the same model.
`threads` applies to its iterations and not to the crossover that may follow
them. `pdlp` runs cuPDLP-C, a primal-dual hybrid gradient method that reads
the matrix only through matrix-vector products. `crossover` controls whether
an interior point method passes its solution to the simplex method to obtain
a vertex.

```python skip="needs a HiGHS built with HiPO"
solution = model.solve(
    options={
        "method": "hipo",
        "newton_system": "augmented",
        "crossover": "off",
        "threads": 8,
    }
)
```

Gurobi and Mosek support `crossover` and `method` up to `barrier`.
`newton_system`, `pdlp_tol`, `hipo` and `pdlp` are specific to HiGHS. Passing
one of them to another solver raises and reports the option. The
[solvers reference](/reference/solvers) records the same rule.

## Crossover

An interior point method stops at a point inside the feasible region,
within tolerance of the optimum on every constraint. Crossover moves that
point to a vertex with the simplex method. A basis, an exact active set and
duals at a vertex require that move. Crossover runs serially, and on a large
model it can cost more than the interior point iterations before it.

`crossover="off"` returns the interior point as the solution. Primals and
duals are read the same way. A constraint that is tight at the optimum may be
a tolerance away from equality, and a variable at a bound may be a tolerance
inside it. The interior point is sufficient where the solution is read as
quantities and prices and not as a basis.

## PDLP on a GPU

A first-order method computes no factorization. Its memory is the matrix in
two orientations, one for each product, plus working vectors of the row and
column dimensions. That memory grows linearly with the problem, and a problem
too large for a factorization still fits.

**Where the memory goes.** HiGHS presolves on the CPU, in host memory,
before any method runs: the original LP and its reduced form are both stored
there while presolve runs, and that is the peak on the host. The reduced
problem is transferred to the GPU. It is smaller than the problem as
declared: presolve removes the rows that bound a single column and the
columns it can fix. The GPU therefore needs memory for the reduced matrix
twice, the working vectors, and the buffers of the sparse kernels, and
nothing else. The host keeps the original and the reduced LP throughout, and
postsolve maps the solution back through them.

A HiGHS built without CUDA runs the same method on the CPU. The solver log,
requested with `log=True`, reports the device the method runs on.

**The tolerance.** `pdlp_tol` is the relative tolerance at which PDLP
stops: the duality gap and the primal and dual residuals, each relative to
the scale of the problem, must all fall below it. A looser tolerance stops
sooner and returns a point farther from the optimum and from feasibility. A
tighter tolerance costs more iterations, and each iteration is a pass over
the matrix. The iterations are cheap and numerous, and the tolerance controls
both the solve time and the accuracy of the solution.

HiGHS checks the point PDLP returns against its own `feasibility_tol` and
`optimality_tol` after postsolve. That check is stricter than the criterion
of PDLP. PDLP measures its residuals relative to the scaled problem it
iterates on. Postsolve maps the point back onto the original rows, where a
residual that passed can exceed the tolerance. A point that meets `pdlp_tol`
and fails the HiGHS check reports the status HiGHS calls unknown. `nimopt`
raises on that status and reports it, and it reads no values from that point.
On a model of any size the default `pdlp_tol` is usually too loose for HiGHS
to accept the point. The two agree at a tolerance one to two orders
tighter. Tighten `pdlp_tol` until the point passes, or loosen the feasibility
and optimality tolerances to the accuracy the model requires.

PDLP reports two iterates in its log, the running average marked `[A]` and
the last marked `[L]`, and stops on whichever meets the tolerance first.
Progress is not monotone: the gap can close and open again as the method
restarts. The method returns the last row of the log, not the best row.

```python skip="needs a HiGHS built with CUDA to run on a GPU"
solution = model.solve(options={"method": "pdlp", "pdlp_tol": 1e-8, "log": True})
```

## Installing a HiGHS with HiPO and GPU support

HiGHS stores the HiPO orderings and its BLAS in a separate library, `libhighs_extras`. The `highspy` wheel on PyPI contains HiPO and contains neither that library nor CUDA. Given `hipo`, a HiGHS without the library logs an error and runs simplex. `nimopt` raises instead and reports the missing library. Given `pdlp`, a HiGHS without CUDA runs the method on the CPU.

**HiPO.** The PyPI package `highspy-extras` contains the library. `highspy` 1.16.0 loads it from the directory of that package at import, and only where both packages have the same version.

```bash
uv pip install --python <venv>/bin/python "highspy[extras]"
```

**The GPU.** PDLP runs on a GPU with a `highspy` built from the HiGHS repository at the tag of the release, with a CUDA toolkit in the environment and the compute capability of the card. Pass the prefix of the conda environment through the `CMAKE_PREFIX_PATH` environment variable, not through `CMAKE_ARGS`. `CMAKE_ARGS` replaces the path the Python build adds for pybind11.

```bash
CMAKE_PREFIX_PATH=$CONDA_PREFIX CUDACXX=$CONDA_PREFIX/bin/nvcc CUDAToolkit_ROOT=$CONDA_PREFIX \
CMAKE_ARGS="-DCUPDLP_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DBLA_VENDOR=OpenBLAS" \
  uv build --wheel --python <venv>/bin/python -o dist .
```

The GPU wheel links the CUDA runtime, cuBLAS and cuSPARSE, and the conda environment that provides them must remain installed. The `libhighs` and `libcudalin` of the wheel require the `lib` directory of that environment in their own run path. `patchelf --set-rpath` sets it on the unpacked wheel before the wheel is packed again. The run path of the extension module alone does not resolve the CUDA libraries.

```bash
uv pip install --python <venv>/bin/python dist/highspy-1.16.0-*.whl "highspy-extras==1.16.0"
```

A solve with `method="hipo"` verifies the HiPO installation. A HiGHS without the extras library raises before anything is solved, and a HiGHS with the library returns the optimum. A solve with `method="pdlp"` and `log=True` on a GPU build prints the CUDA device in its log.
