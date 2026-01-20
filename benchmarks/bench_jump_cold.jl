using JuMP
using HiGHS
using Random

# Time everything from the start (after package load)
total_start = time_ns()

Random.seed!(42)
n_src, n_dst = 300, 300

supply_vals = 100 .+ 900 .* rand(n_src)
demand_vals = 50 .+ 150 .* rand(n_dst)
total_supply = sum(supply_vals)
total_demand = sum(demand_vals)
demand_vals = demand_vals .* (total_supply / total_demand * 0.9)
cost_vals = 1 .+ 9 .* rand(n_src, n_dst)

t0 = time_ns()

model = Model(HiGHS.Optimizer)
set_silent(model)

@variable(model, x[1:n_src, 1:n_dst] >= 0)
@objective(model, Min, sum(cost_vals[i, j] * x[i, j] for i in 1:n_src, j in 1:n_dst))
@constraint(model, supply[i=1:n_src], sum(x[i, j] for j in 1:n_dst) <= supply_vals[i])
@constraint(model, demand[j=1:n_dst], sum(x[i, j] for i in 1:n_src) >= demand_vals[j])

t1 = time_ns()
optimize!(model)
t2 = time_ns()

build_ms = (t1 - t0) / 1e6
solve_ms = (t2 - t1) / 1e6
total_from_code_ms = (t2 - t0) / 1e6
total_with_jit_ms = (t2 - total_start) / 1e6

println("300x300 (90k vars) - COLD START (includes JIT):")
println("  Build (with JIT): $(round(build_ms, digits=1)) ms")
println("  Solve:            $(round(solve_ms, digits=1)) ms")
println("  Total (code):     $(round(total_from_code_ms, digits=1)) ms")
println("  Total (with JIT): $(round(total_with_jit_ms, digits=1)) ms")
println("  Obj:              $(round(objective_value(model), digits=2))")
