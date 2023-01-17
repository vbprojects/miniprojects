using JuMP, MLJ, Tables, LinearAlgebra, HiGHS, Plots

rbf(x, y) = exp(-norm(x - y)^2 / 10)
sigmoid(x) = 1 / (1 + exp(-x))

N = 1000
D = make_moons(N)
X = D[1] |> Tables.matrix |> eachrow .|> collect
Y = D[2] .== 1 .|> Float64
d = length(X[1])
w = zeros(2)
S = ones(N)

K = [rbf(X[i], X[j]) for i in 1:N, j in 1:N] |> eachrow .|> collect
C = 1.0
# @variable(model, X[1:N, 1:d])
# @variable(model, Y[1:N])

begin
    model = Model(HiGHS.Optimizer)
    @variable(model, w[1:N])
    @variable(model, b)
    @variable(model, S[1:N])

    for i in 1:N
        @constraint(model, Y[i] * (w' * K[i] + b) + C * S[i] >= 1)
    end

    @objective(model, Min, sum(S))

    optimize!(model)
end
w = value.(w)

function predict(x)
    return w' * [rbf(x, K[i]) for i in 1:N] + value(b)
end
findall(==(0), Y)
pred = predict.(K) .> .5
co = Y .!= pred
Xv = D[1] |> Tables.matrix
scatter(Xv[:, 1], Xv[:, 2], group = co)