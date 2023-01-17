using Zygote, Plots, LinearAlgebra, MLJ

loss(w, K, Y) = 1/2 * sum(w .^ 2) + sum(max.(0, 1 .- Y .* (K * w)))

N = 200
hN = N // 2 |> Int64
c = collect
rs(s) = x -> reshape(x, s)

X1 = randn(hN, 2) .+ 2
X2 = randn(hN, 2) .- 2

X = [X1 ; X2]
Y = [ones(hN) ; -ones(hN)]

scatter(X[:, 1], X[:, 2], c = Y .|> Int64, legend = false)

w = zeros(2)
for _ in 1:100
    w -= 0.01 * gradient(w -> loss(w, X, Y), w)[1]
end

sum(Y .== sign.(X * w)) / N

rbf(x, y, σ) = exp(-norm(x - y)^2 / (2 * σ^2))

K = [rbf(X[i, :], X[j, :], 1) for i in 1:N, j in 1:N]

w = zeros(N)
for _ in 1:100
    w -= 0.01 * gradient(w -> loss(w, K, Y), w)[1]
end

sum(Y .== sign.(K * w)) / N

D = make_circles(N, noise=.05, factor = .5)
begin
    Y = D[2] |> Vector{Float64}
    Y = 2 .* Y .- 1
end
X = [D[1][:x1] D[1][:x2]]

scatter(X[:, 1], X[:, 2], c = Y .|> Int64, legend = false)

w = zeros(2)
for _ in 1:1000
    w -= 0.01 * gradient(w -> loss(w, X, Y), w)[1]
end

sum(Y .== sign.(X * w)) / N

negc(x) = x == 1 ? :red : :green
neg = (Y .!= sign.(X * w)) .|> negc
scatter(X[:, 1], X[:, 2], c = neg, legend = false)

K = [rbf(X[i, :], X[j, :], .1) for i in 1:N, j in 1:N]

w = zeros(N)
for _ in 1:1000
    w -= 0.01 * gradient(w -> loss(w, K, Y), w)[1]
end

sum(Y .== sign.(K * w)) / N
neg = (Y .!= sign.(K * w)) .|> negc 
scatter(X[:, 1], X[:, 2], c = neg, legend = false)

histogram(w, bins = 20)

sv = sortperm(abs.(w))[1:150]

Ks = [rbf(X[i, :], X[j, :], .1) for i in 1:N, j in sv]
ws = w[sv]
sum(Y .== sign.(Ks * ws)) / N
neg = (Y .!= sign.(Ks * ws)) .|> negc 
scatter(X[:, 1], X[:, 2], c = neg, legend = false)