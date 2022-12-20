using Plots, LinearAlgebra, StatsBase

X = -3:.1:4 |> collect
f(x) = sinc(x ^ 2) ^ 3 + cosc(x) .+ 3 .+ x
Y = f.(X) .+ rand(length(X)) .* .5

function kmeansinit(X, Y, n)
    reldist(X, i) = [norm(x .- X[i]) for x in X] + broadcast.(norm, Y, Y[i])
    xs = zeros(Int, n)
    xs[1] = rand(1:length(Y))
    xd = Dict(1 => reldist(X, xs[1]), 0 => ones(length(Y)))
    for i in 2:n
        xs[i] = sample(1:length(X), Weights(xd[i-1] .+ xd[i-2]))
        xd[i] = reldist(X, xs[i])
    end
    return xs
end

kmeansinit(X, Y, 100)

rbf(x, y, s) = exp(-1.0 * norm(x .- y) ^ 2 / s)
poly(x, y, s) = (x' * y + 1) ^ s
constant(x, y) = [x; 1]'*[y; 1]
# rbf(x:, y::Float, s) = exp(-1.0 * (x - y) ^ 2 / s)

function gKn(X, Y, af, n)
    inds = kmeansinit(X, Y, n)
    Xinds = X[inds]
    K = [af(x, c) for x in X, c in Xinds]
    return (K, inds)
end

function gnX(Xinds, af, nX)
    K = [af(x, c) for x in nX, c in Xinds]
    return K
end

function rbfmachine(layers, X, Y)
    inds = Dict()
    cX = X
    for (i, (n, s)) in enumerate(layers)
        (K, ind) = gKn(cX, Y, s, n)
        inds[i] = cX[ind]
        cX = K |> eachrow .|> collect
    end
    alpha = pinv(reduce(hcat, cX)')*Y
    (alpha, inds, [s for (_, s) in layers])
end

function rbfpred(X, alpha, inds, S)
    cX = X
    for i in 1:length(inds)
        Xind = inds[i]
        K = gnX(Xind, S[i], cX)
        cX = K |> eachrow .|> collect
    end
    reduce(hcat, cX)' * alpha
end

sink(x, y, s) = sin(-1 * norm(x .- y) .^ 2 / s)


rbfl(n, s) = (n, (x, y) -> rbf(x, y, s))
polyl(n, s) = (n, (x, y) -> poly(x, y, s))
sinkl(n, s) = (n, (x, y) -> sink(x, y, s))
sigl(n, g, r) = (n, (x, y) -> tanh(g * x'*y - r))
constantl(n) = (n, constant)

network = [
    polyl(5, 1),
]

alpha, inds, S = rbfmachine(network, X, Y)

nX = -10:.01:10 |> collect
Yh = rbfpred(nX, alpha, inds, S)

plot(nX, Yh, label="rbf", legend = :outertopright, xlims=(-10, 10), ylims=(-2, 10))
plot!(x -> f(x), label="f(x)")
scatter!(X, Y, label="data", markersize = 2)
sum(abs2, mean(f.(nX) .- Yh)), sum(abs, mean(f.(X) .- rbfpred(X, alpha, inds, S)))