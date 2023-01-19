using JuMP, MLJ, Tables, LinearAlgebra, HiGHS, Plots, COSMO

rbf(x, y) = exp(-norm(x - y)^2 / .1)
sigmoid(x) = 1 / (1 + exp(-x))

N = 1000
D = make_moons(N)
X = D[1] |> Tables.matrix |> eachrow .|> collect
Y = (D[2] .== 1 .|> Float64) .* 2 .- 1
d = length(X[1])
w = zeros(2)
S = ones(N)
lam = 10

begin
    model = JuMP.Model(COSMO.Optimizer)
    @variable(model, 0 <= c[1:N] <= 1/(2 * N * lam))
    @constraint(model, sum(c[i] * Y[i] for i in 1:N) == 0)
    @objective(model, Max, -1/2 * sum(Y[i] * c[i] * rbf(X[i], X[j]) * Y[j] * c[j] for i in 1:N, j in 1:N) + sum(c[i] for i in 1:N))
    optimize!(model)
end

function predict(z)
    C = value.(c)
    sum(C[i] * Y[i] * rbf(X[i], z) for i in 1:N)
end

Xv = D[1] |> Tables.matrix
wrong = (x -> x == 1 ? "wrong" : "right").(sign.(predict.(X)) .!= Y) 
scatter(Xv[:, 1], Xv[:, 2], group = wrong)

value.(c) |> sort |> reverse |> plot