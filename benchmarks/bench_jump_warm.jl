using JuMP
using HiGHS
using Random

println("=" ^ 70)
println("WARM BENCHMARK: Multi-Commodity Transport")  
println("=" ^ 70)

# Warm up
println("\nWarming up JIT...")
for _ in 1:2
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:10, 1:10, 1:5] >= 0)
    @objective(model, Min, sum(x))
    @constraint(model, [i=1:10, k=1:5], sum(x[i,j,k] for j in 1:10) <= 100)
    optimize!(model)
end
println("JIT warm.\n")

function bench_multicommodity(n_src, n_dst, n_comm)
    Random.seed!(42)
    
    supply = 100 .+ 900 .* rand(n_src, n_comm)
    demand = 50 .+ 150 .* rand(n_dst, n_comm)
    # Scale demand
    for k in 1:n_comm
        demand[:, k] .*= sum(supply[:, k]) / sum(demand[:, k]) * 0.9
    end
    cost = 1 .+ 9 .* rand(n_src, n_dst, n_comm)
    arc_cap = 500 .+ 500 .* rand(n_src, n_dst)  # shared capacity
    
    t0 = time_ns()
    
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    @variable(model, x[1:n_src, 1:n_dst, 1:n_comm] >= 0)
    
    @objective(model, Min, sum(cost[i,j,k] * x[i,j,k] 
        for i in 1:n_src, j in 1:n_dst, k in 1:n_comm))
    
    # Supply constraints per commodity
    @constraint(model, sup[i=1:n_src, k=1:n_comm],
        sum(x[i,j,k] for j in 1:n_dst) <= supply[i,k])
    
    # Demand constraints per commodity  
    @constraint(model, dem[j=1:n_dst, k=1:n_comm],
        sum(x[i,j,k] for i in 1:n_src) >= demand[j,k])
    
    # Arc capacity (sum over commodities)
    @constraint(model, cap[i=1:n_src, j=1:n_dst],
        sum(x[i,j,k] for k in 1:n_comm) <= arc_cap[i,j])
    
    t1 = time_ns()
    optimize!(model)
    t2 = time_ns()
    
    n_vars = n_src * n_dst * n_comm
    n_cons = n_src * n_comm + n_dst * n_comm + n_src * n_dst
    
    return (
        n_vars = n_vars,
        n_cons = n_cons,
        build = (t1 - t0) / 1e6,
        solve = (t2 - t1) / 1e6,
        total = (t2 - t0) / 1e6,
        obj = objective_value(model)
    )
end

cases = [
    (100, 100, 10),   # 100k vars
    (200, 200, 10),   # 400k vars
    (300, 300, 10),   # 900k vars
    (200, 200, 50),   # 2M vars
    (300, 300, 30),   # 2.7M vars
]

println("-" ^ 70)
println("Case                 Vars         Cons        Build      Solve      Total")
println("-" ^ 70)

for (n_src, n_dst, n_comm) in cases
    results = [bench_multicommodity(n_src, n_dst, n_comm) for _ in 1:3]
    r = results[sortperm([x.total for x in results])[2]]
    
    label = "$(n_src)x$(n_dst)x$(n_comm)"
    vars_str = r.n_vars >= 1_000_000 ? "$(round(r.n_vars/1e6, digits=1))M" : "$(div(r.n_vars, 1000))k"
    cons_str = r.n_cons >= 1_000_000 ? "$(round(r.n_cons/1e6, digits=1))M" : "$(div(r.n_cons, 1000))k"
    
    println("$(rpad(label, 16)) $(lpad(vars_str, 10))  $(lpad(cons_str, 10))  $(lpad(round(Int, r.build), 8)) ms $(lpad(round(Int, r.solve), 8)) ms $(lpad(round(Int, r.total), 8)) ms")
end
println("-" ^ 70)
