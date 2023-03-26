using Plots, Zygote, Turing, Optim, DataFrames

kern(x) = [xi^i for i in 0:3 for xi in x]
c = collect
X = -1:.01:1 |> c
Ϛ(x, α, β) = β'*sin.(α .* x)
Ϛ(x, α, β, ϕ) = β'*sin.(α .* ϕ(x))
Y = [Ϛ(X[i], [1,2,3,4], [1,2,3,4], kern) for i in axes(X, 1)]
loss(X, Y, α, β) = sum((Ϛ(X[i, :], α, β) - Y[i])^2 for i in axes(X, 1))
gr()
plot(X, Y)
nk(x) = x

α = rand(4)
β = rand(4)
K = reduce(hcat, [kern(X[i]) for i in axes(X, 1)])' |> c

loss(K, Y, rand(4), rand(4))

op = optimize(w -> loss(K, Y, w[1:4], w[5:end]), rand(8), ParticleSwarm(n_particles=100))

a = op.minimizer[1:4]
b = op.minimizer[5:end]

plot(X .* 10, Y)

plot(X, [Ϛ(X[i], a, b, kern) for i in axes(X, 1)], label = "fit")
plot!(X, Y, label = "truth")