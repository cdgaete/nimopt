using JuMP
using HiGHS
using Random

println("=" ^ 70)
println("WARM BENCHMARK: Complex Energy Model")
println("=" ^ 70)

# Warm up JIT with small model
println("\nWarming up JIT...")
let
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:10, 1:10] >= 0)
    @objective(model, Min, sum(x))
    @constraint(model, [i=1:10], sum(x[i,j] for j in 1:10) <= 100)
    optimize!(model)
end
println("JIT warm.")

function bench_energy_model(n_time, n_gen, n_node; verbose=false)
    Random.seed!(42)
    
    # Parameters
    demand = 50 .+ 50 .* rand(n_time, n_node)
    gen_cost = 10 .+ 40 .* rand(n_gen)
    gen_cap = 100 .+ 200 .* rand(n_gen)
    gen_node = rand(1:n_node, n_gen)  # which node each generator is at
    line_cap = 50 .+ 50 .* rand(n_node, n_node)
    
    t0 = time_ns()
    
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    # Variables
    @variable(model, gen[1:n_time, 1:n_gen] >= 0)  # generation
    @variable(model, flow[1:n_time, 1:n_node, 1:n_node])  # power flow
    @variable(model, unmet[1:n_time, 1:n_node] >= 0)  # unmet demand
    
    # Objective: minimize cost + penalty for unmet demand
    @objective(model, Min, 
        sum(gen_cost[g] * gen[t, g] for t in 1:n_time, g in 1:n_gen) +
        1000 * sum(unmet[t, n] for t in 1:n_time, n in 1:n_node)
    )
    
    # Generation capacity
    @constraint(model, cap[t=1:n_time, g=1:n_gen], 
        gen[t, g] <= gen_cap[g])
    
    # Flow limits
    @constraint(model, flow_cap[t=1:n_time, i=1:n_node, j=1:n_node],
        flow[t, i, j] <= line_cap[i, j])
    @constraint(model, flow_cap_neg[t=1:n_time, i=1:n_node, j=1:n_node],
        flow[t, i, j] >= -line_cap[i, j])
    
    # Node balance: generation + inflow - outflow + unmet = demand
    @constraint(model, balance[t=1:n_time, n=1:n_node],
        sum(gen[t, g] for g in 1:n_gen if gen_node[g] == n) +
        sum(flow[t, i, n] for i in 1:n_node if i != n) -
        sum(flow[t, n, j] for j in 1:n_node if j != n) +
        unmet[t, n] == demand[t, n]
    )
    
    t1 = time_ns()
    optimize!(model)
    t2 = time_ns()
    
    n_vars = n_time * n_gen + n_time * n_node * n_node + n_time * n_node
    n_cons = n_time * n_gen + 2 * n_time * n_node * n_node + n_time * n_node
    
    build_ms = (t1 - t0) / 1e6
    solve_ms = (t2 - t1) / 1e6
    total_ms = (t2 - t0) / 1e6
    obj = objective_value(model)
    
    if verbose
        println("  Variables:    $n_vars")
        println("  Constraints:  $n_cons")
        println("  Build:        $(round(build_ms, digits=1)) ms")
        println("  Solve:        $(round(solve_ms, digits=1)) ms")
        println("  Total:        $(round(total_ms, digits=1)) ms")
        println("  Objective:    $(round(obj, digits=2))")
    end
    
    return (n_vars=n_vars, n_cons=n_cons, build=build_ms, solve=solve_ms, total=total_ms, obj=obj)
end

# Test cases
cases = [
    (24, 50, 10),    # Small: 24 hours, 50 generators, 10 nodes
    (168, 100, 20),  # Medium: 1 week, 100 generators, 20 nodes  
    (720, 200, 30),  # Large: 1 month, 200 generators, 30 nodes
    (8760, 100, 10), # XL: 1 year hourly, 100 generators, 10 nodes
]

println("\n" * "-" ^ 70)
println("Case                    Vars        Cons        Build      Solve      Total")
println("-" ^ 70)

for (n_time, n_gen, n_node) in cases
    # Run 3 times, take median
    results = [bench_energy_model(n_time, n_gen, n_node) for _ in 1:3]
    totals = [r.total for r in results]
    r = results[sortperm(totals)[2]]
    
    label = "$(n_time)t x $(n_gen)g x $(n_node)n"
    println("$(rpad(label, 20)) $(lpad(r.n_vars, 10))  $(lpad(r.n_cons, 10))  $(lpad(round(r.build, digits=0), 8)) ms $(lpad(round(r.solve, digits=0), 8)) ms $(lpad(round(r.total, digits=0), 8)) ms")
end

println("-" ^ 70)
println("\nDetailed XL case (8760 x 100 x 10):")
bench_energy_model(8760, 100, 10, verbose=true)
