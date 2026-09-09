---
title: Interior point and first-order methods
description: HiGHS's simplex, IPX, HiPO and PDLP through nimopt's options, what each holds in memory, and how to install a HiGHS that carries HiPO and a GPU.
---

# Interior point and first-order methods

`method=` names the algorithm HiGHS runs on the assembled matrix. Four
values name one each, and three further options shape the interior point
and first-order methods. Every setting on this page runs in the test
suite against the bundled models, and a solve carrying it reports the
optimum the model has; which method is fastest or smallest on a given
model is the model's to show.

| Option | Choices | Applies to |
| --- | --- | --- |
| `method` | `choose`, `simplex`, `barrier`, `hipo`, `pdlp` | every solve |
| `newton_system` | `choose`, `augmented`, `normaleq` | `hipo` |
| `crossover` | `choose`, `off`, `on` | `barrier` and `hipo` |
| `pdlp_tol` | a relative tolerance | `pdlp` |

`barrier` runs IPX, HiGHS's interior point method with a preconditioned
conjugate gradient at its core. `hipo` runs HiPO, an interior point method
built on a direct factorisation of the Newton system, parallel across the
elimination tree; `newton_system` chooses between the augmented system and
the normal equations, and `choose` leaves that to the solver. HiPO holds
that factorisation in memory, so it needs more of it than IPX on the same
model, and `threads` reaches its iterations but not the crossover that may
follow them. `pdlp` runs cuPDLP-C, a primal-dual hybrid gradient method that
touches the matrix only through matrix-vector products. `crossover` decides
whether an interior point method hands its solution to the simplex method
to reach a vertex.

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

Gurobi and Mosek carry `crossover` and `method` up to `barrier`.
`newton_system`, `pdlp_tol`, `hipo` and `pdlp` are HiGHS's, and asking either
of the others for one of them is refused by name, as the
[solvers reference](/reference/solvers) states.

## Crossover

An interior point method stops at a point inside the feasible region,
within tolerance of the optimum on every constraint. Crossover moves that
point to a vertex with the simplex method, which is what a basis, an exact
active set and duals at a vertex require. It runs serially, and on a large
model it can cost more than the interior point iterations before it.

`crossover="off"` returns the interior point as the solution. Primals and
duals are read back the same way; what changes is that a constraint holding
with equality at the optimum may sit a tolerance away from it, and a
variable at a bound may sit a tolerance inside it. A model whose answer is
read as quantities and prices, rather than as a basis, is served by the
interior point.

## PDLP on a GPU

A first-order method holds no factorisation. Its memory is the matrix in
two orientations, one for each product, and a set of working vectors of the
row and column dimensions, so it grows linearly with the problem and a
problem too large for a factorisation still fits.

**Where the memory goes.** HiGHS presolves on the CPU, in host memory,
before any method runs: the original LP and its reduced form are both held
there while presolve works, which is the host's peak. The reduced problem is
what moves to the GPU, and it is smaller than the problem stated, because
presolve removes the rows that only bound a single column and the columns
presolve can fix. The GPU therefore needs memory for the reduced matrix
twice, the working vectors, and the sparse kernels' buffers, and nothing
else. The host keeps the original and the reduced LP throughout, and
postsolve maps the answer back through them.

A HiGHS built without CUDA runs the same method on the CPU. The solver's
log, asked for with `log=True`, names the device it runs on.

**The tolerance.** `pdlp_tol` is the relative tolerance at which PDLP
stops: the duality gap and the primal and dual residuals, each relative to
the scale of the problem, must all fall below it. A looser tolerance stops
sooner and returns a point farther from the optimum and farther from
feasibility; a tighter one costs more iterations, and each iteration is a
pass over the matrix. The method's iterations are cheap and numerous, so
the tolerance is the lever on both the time and the quality of the answer.

HiGHS checks the point PDLP returns against its own `feasibility_tol` and
`optimality_tol` after postsolve, and that check is stricter than PDLP's
own criterion: PDLP measures its residuals relative to the scaled problem
it iterates on, and postsolve maps the point back onto the original rows,
where a residual that passed can exceed the tolerance. A point that meets
`pdlp_tol` and misses HiGHS's check reports the status HiGHS calls unknown,
and `nimopt` refuses that status rather than reading values from it: the
solve raises, naming the status. On a model of any size, the default
`pdlp_tol` is therefore usually not enough for HiGHS to accept the point,
and a tolerance one to two orders tighter is where the two agree.
Tightening `pdlp_tol` until the point passes, or loosening the feasibility
and optimality tolerances to what the model needs, are the two ways
through.

PDLP reports two iterates in its log, the running average marked `[A]` and
the last marked `[L]`, and stops on whichever meets the tolerance first.
Progress is not monotone: the gap can close and open again as the method
restarts, so the last row of the log, not the best one, is what it returns.

```python skip="needs a HiGHS built with CUDA to run on a GPU"
solution = model.solve(options={"method": "pdlp", "pdlp_tol": 1e-8, "log": True})
```

## Installing a HiGHS that carries HiPO and a GPU

HiGHS 1.15.1 keeps HiPO's orderings and its BLAS in a library of its own,
`libhighs_extras`, which `libhighs` loads at run time by name. The `highspy`
wheel on PyPI and the conda-forge package ship without that library and
without CUDA. Asked for `hipo`, such a HiGHS logs an error and runs simplex;
`nimopt` refuses the request instead, naming the missing library. Asked for
`pdlp`, it runs the method on the CPU.

Both come from a source build of the same version, with three pieces
installed separately:

1. **The extras library**, from the `extern` directory of the HiGHS
   repository, which is a CMake project of its own. It builds METIS, AMD
   and RCM from the tree and links a BLAS; OpenBLAS from a conda
   environment serves.

   ```bash
   cmake -S extern -B build-extras -G Ninja -DCMAKE_BUILD_TYPE=Release \
     -DHIPO=ON -DBLA_VENDOR=OpenBLAS
   cmake --build build-extras
   ```

2. **The wheel**, from the repository root, where HiGHS's own build
   configuration turns HiPO on. The conda environment's prefix is given
   through the `CMAKE_PREFIX_PATH` environment variable, not through
   `CMAKE_ARGS`, because the latter replaces the path the Python build adds
   for pybind11.

   ```bash
   CMAKE_PREFIX_PATH=$CONDA_PREFIX CMAKE_ARGS="-DBLA_VENDOR=OpenBLAS" \
     uv build --wheel --python <venv>/bin/python -o dist .
   ```

   For the GPU, the same command with a CUDA toolkit in the environment and
   the card's compute capability:

   ```bash
   CMAKE_PREFIX_PATH=$CONDA_PREFIX CUDACXX=$CONDA_PREFIX/bin/nvcc CUDAToolkit_ROOT=$CONDA_PREFIX \
   CMAKE_ARGS="-DCUPDLP_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DBLA_VENDOR=OpenBLAS" \
     uv build --wheel --python <venv>/bin/python -o dist .
   ```

3. **The placement.** The wheel's `libhighs` searches its own directory
   for the extras library, so the built `libhighs_extras.so` goes into the
   `highspy` package directory of the environment the wheel is installed
   in. The extras library links the BLAS it was built against, and the GPU
   wheel links the CUDA runtime, cuBLAS and cuSPARSE, so the conda
   environment that provided them stays. The GPU wheel's `libhighs` and
   `libcudalin` must name that environment's `lib` in their own run path,
   which `patchelf --set-rpath` sets on the unpacked wheel before it is
   packed again; the extension module's run path does not reach the CUDA
   libraries through them.

   ```bash
   uv pip install --python <venv>/bin/python dist/highspy-1.15.1-*.whl
   cp build-extras/libhighs_extras.so <venv>/lib/python3.13/site-packages/highspy/
   ```

The check that HiPO is in place is a solve asking for it: a HiGHS without
the extras library is refused before anything is solved, and one with it
returns the optimum.
