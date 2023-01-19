using JuMP, MLJ, Tables, LinearAlgebra, Plots, COSMO

rbf(x, y) = exp(-norm(x - y)^2 / .1)
sigmoid(x) = 1 / (1 + exp(-x))

N = 1000
D = make_moons(N)
X = D[1] |> Tables.matrix |> eachrow .|> collect
Y = (D[2] .== 1 .|> Float64) .* 2 .- 1
d = length(X[1])
w = zeros(2)
S = ones(N)
lam = .1

begin
    model = JuMP.Model(COSMO.Optimizer)
    @variable(model, 0 <= c[1:N] <= 1/(2 * N * lam))
    @constraint(model, sum(c[i] * Y[i] for i in 1:N) == 0)
    @objective(model, Max, -1/2 * sum(Y[i] * c[i] * ρ(X[i], X[j]) * Y[j] * c[j] for i in 1:N, j in 1:N) + sum(c[i] for i in 1:N))
    optimize!(model)
end

ρ(x, y) = rbf(x, y)

function predict(Z, I)
    C = value.(c)
    pYs = zeros(length(Z))
    for j in 1:length(Z)
        pYs[j] = sum(C[i] * Y[i] * ρ(X[i], Z[j]) for i in I) |> sign
    end
    return pYs
end

Xv = D[1] |> Tables.matrix
inds = sortperm(value.(c))
wrong = (x -> x == 1 ? "wrong" : "right").(sign.(predict(X,inds)) .!= Y) 
scatter(Xv[:, 1], Xv[:, 2], group = wrong)
value.(c) |> sort |> reverse |> plot

# scatter(Xv[:, 1], Xv[:, 2], group = Y, legend = false)
