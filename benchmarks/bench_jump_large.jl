using JuMP
using HiGHS
using Random

function bench_transport(n_src, n_dst)
    Random.seed!(42)
    
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
    
    return (build=(t1-t0)/1e6, solve=(t2-t1)/1e6, total=(t2-t0)/1e6, obj=objective_value(model))
end

# Warm up
println("Warming up JIT...")
bench_transport(10, 10)
bench_transport(30, 30)
bench_transport(100, 100)
println()

println("=" ^ 70)
println("JuMP WARM Benchmark - Large Scale")
println("=" ^ 70)
println()

# Large models
sizes = [
    (500, 500),
    (700, 700),
    (1000, 1000),
    (1000, 2000),
    (1500, 1500),
    (2000, 2000),
]

for (n_src, n_dst) in sizes
    n_vars = n_src * n_dst
    n_cons = n_src + n_dst
    
    # Single run for large models
    result = bench_transport(n_src, n_dst)
    
    println("$(n_src)x$(n_dst): $(n_vars) vars, $(n_cons) cons")
    println("  Build: $(round(result.build, digits=0)) ms")
    println("  Solve: $(round(result.solve, digits=0)) ms")
    println("  Total: $(round(result.total, digits=0)) ms")
    println()
end
