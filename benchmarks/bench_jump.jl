using JuMP
using HiGHS
using Random

function bench_transport(n_src, n_dst; verbose=false)
    Random.seed!(42)
    
    supply_vals = 100 .+ 900 .* rand(n_src)
    demand_vals = 50 .+ 150 .* rand(n_dst)
    total_supply = sum(supply_vals)
    total_demand = sum(demand_vals)
    demand_vals = demand_vals .* (total_supply / total_demand * 0.9)
    cost_vals = 1 .+ 9 .* rand(n_src, n_dst)
    
    # Build
    t0 = time_ns()
    
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    @variable(model, x[1:n_src, 1:n_dst] >= 0)
    
    @objective(model, Min, sum(cost_vals[i, j] * x[i, j] for i in 1:n_src, j in 1:n_dst))
    
    @constraint(model, supply[i=1:n_src], sum(x[i, j] for j in 1:n_dst) <= supply_vals[i])
    @constraint(model, demand[j=1:n_dst], sum(x[i, j] for i in 1:n_src) >= demand_vals[j])
    
    t1 = time_ns()
    
    # Solve
    optimize!(model)
    t2 = time_ns()
    
    build_ms = (t1 - t0) / 1e6
    solve_ms = (t2 - t1) / 1e6
    total_ms = (t2 - t0) / 1e6
    obj = objective_value(model)
    
    if verbose
        println("  Build: $(round(build_ms, digits=1)) ms")
        println("  Solve: $(round(solve_ms, digits=1)) ms")
        println("  Total: $(round(total_ms, digits=1)) ms")
        println("  Obj:   $(round(obj, digits=2))")
    end
    
    return (build=build_ms, solve=solve_ms, total=total_ms, obj=obj)
end

# Warm up JIT
println("Warming up JIT...")
bench_transport(10, 10)
bench_transport(10, 10)

println()
println("=" ^ 60)
println("JuMP/HiGHS Benchmark")
println("=" ^ 60)
println()

sizes = [(30, 30), (100, 100), (200, 200), (300, 300), (400, 400)]

for (n_src, n_dst) in sizes
    n_vars = n_src * n_dst
    
    # Run 3 times, take median
    times = [bench_transport(n_src, n_dst) for _ in 1:3]
    totals = [t.total for t in times]
    median_idx = sortperm(totals)[2]
    result = times[median_idx]
    
    println("$(n_src)x$(n_dst) ($(n_vars) vars): $(round(result.total, digits=1)) ms")
end

println()
println("2M variables benchmark:")
println()
result = bench_transport(1000, 2000, verbose=true)
