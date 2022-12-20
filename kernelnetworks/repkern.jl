using Plots, LinearAlgebra, StatsBase

X = sort(randn(1000) .* 10)
Y = sinc.(X) .- cos.(X) .+ rand(length(X)) .* .5


sumabs(x) = sum(abs2, x)


scatter(X, Y, label = "Data", legend = :outertopright, markersize = 2) #xlims = (-10, 10), ylims = (-2, 2))
rbfinds1 = randindinit(1000, Y)
rbf(x) = exp.(-1.0 .* x .^ 2 ./ 10)


X1 = reduce(hcat, broadcast.(-, X[rbfinds1], (X,)) .|> rbf)

plot!(X, X1*pinv(X1)*Y)
# scatter!(X[rbfinds1], Y[rbfinds1], label = "samples")

rbfinds2 = randindinit(1000, Y)
XT = X1 |> eachrow .|> collect
Xt = [rbf(norm(XT[i] .- XT[rbfinds2][j])^2) for i in 1:length(XT), j in 1:length(rbfinds2)]
plot!(X, Xt*(pinv(Xt)*Y))
# scatter!(X[rbfinds2], Y[rbfinds2], label = "samples2")

XT2 = Xt |> eachrow .|> collect
rbfinds3 = randindinit(1000, Y)
Xt2 = [rbf(norm(XT2[i] .- XT2[rbfinds3][j])^2) for i in 1:length(XT2), j in 1:length(rbfinds3)]
plot!(X, Xt2*(pinv(Xt2)*Y))
# scatter!(X[rbfinds3], Y[rbfinds3], label = "samples3")


Y - X1*pinv(X1)*Y |> sumabs
Y - Xt*(pinv(Xt)*Y) |> sumabs
Y - Xt2*(pinv(Xt2)*Y) |> sumabs

inds1 = randindinit(5, Y)

function randindinit(n,Y)
    distv(Y, y) = broadcast.(norm, Y, y)
    xs = zeros(Int, n)
    xs[1] = rand(1:length(Y))
    xd1 = distv(Y, Y[xs[1]])
    xd = Dict(1 => xd1 .^ 2, 0 => xd1 .^ 2)
    for i in 2:n
        xs[i] = sample(1:length(X), Weights(xd[i-1] .+ xd[i-2]))
        xd[i] = distv(Y, Y[xs[i]]) .^ 2
    end
    return xs
end

function rbfmachine(layers, X, Y)
    iX = 1:length(Y) |> collect
    inds = Dict(1 => randindinit(layers[1], Y))
    Ks = Dict(1 => X)
    for i in 1:length(layers)
        Ks[i+1] = [rbf(norm(Ks[i][j] .- Ks[i][inds[i][k]])^2) for j in 1:length(Ks[i]), k in 1:length(inds[i])]
    end
    Ks
end

rbfmachine([10], X, Y)

layers = [10]
iX = 1:length(Y) |> collect
inds = Dict(1 => randindinit(layers[1], Y))
Ks = Dict(1 => X)
i = 1

rbf(x, y, s) = exp(-1.0 * sum(abs2, x .- y) / s)

# for i in 1:length(layers)
Ks[i+1] = [rbf(norm(Ks[i][j] .- Ks[i][inds[i]][k])^2) for j in 1:length(Ks[i]), k in 1:length(inds[i])]

Ks1 = Ks[i]
Ksa = Ks[i][inds[i]]

(Ix'*y for x in 1:length(Ks1), for y in 1:length(Ksa))