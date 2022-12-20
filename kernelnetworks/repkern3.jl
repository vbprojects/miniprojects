using Plots, LinearAlgebra, StatsBase, FLoops, LambdaFn

X = -2:.01:2 |> collect
f(x) = sinc(x ^ 2) ^ 3 + cosc(x) .+ 3 .+ x
Y = f.(X) .+ rand(length(X)) .* .5

function reldist(X, Y, i)
    m = size(X)[1]
    dist = zeros(m)
    @floop for j in 1:m
        dist[j] = norm(X[j, :] .- X[i, :]) + norm(Y[j] - Y[i])
    end
    return dist
end

reldist(X, Y, 1)

function kmeansinit(X, Y, n)
    m = size(X)[1]
    xs = zeros(Int, n)
    xs[1] = argmax(Y)
    xd0 = ones(m)
    xd1 = reldist(X, Y, xs[1])
    for i in 2:n
        xs[i] = sample(1:m, Weights(xd1 .+ xd0))
        xd0 = xd1
        xd1 = reldist(X, Y, xs[i])
    end
    return xs
end

function gKn(X, Y, af, n)
    inds = kmeansinit(X, Y, n)
    Xinds = X[inds, :]
    K = ka(X, Xinds, af)
    return (K, inds)
end

function ka(X, Xinds, af)
    m = size(X)[1]
    n = size(Xinds)[1]
    K = zeros(m, n)
    for i in 1:m
        for j in 1:n
            K[i, j] = af(X[i, :], Xinds[j, :])
        end
    end
    return K
end

function repkmachine(layers, X, Y)
    inds = Dict()
    cinds = Dict()
    cX = X
    for (i, (n, s)) in enumerate(layers)
        (K, ind) = gKn(cX, Y, s, n)
        inds[i] = cX[ind, :]
        cinds[i] = ind
        cX = K
    end
    alpha = pinv(cX)*Y
    (alpha, inds, [s for (_, s) in layers], cinds)
end

function repkpred(X, alpha, inds, S)
    cX = X
    for i in 1:length(inds)
        Xind = inds[i]
        K = ka(cX,Xind, S[i])
        cX = K
    end
    cX * alpha
end


rbf(x, y, s) = exp(-1.0 * norm(x .- y) ^ 2 / s)
poly(x, y, s) = (x' * y + 1) ^ s
constant(x, y) = [x; 1]'*[y; 1]
sink(x, y, s) = sin(-1 * norm(x .- y) .^ 2 / s)

rbfl(n, s) = (n, (x, y) -> rbf(x, y, s))
polyl(n, s) = (n, (x, y) -> poly(x, y, s))
sinkl(n, s) = (n, (x, y) -> sink(x, y, s))
sigl(n, g, r) = (n, (x, y) -> tanh(g * x'*y - r))

rbfpolyl(n, s, d) = (n, (x, y) -> rbf(x, y, s) * poly(x, y, d))

constantl(n) = (n, constant)

iter(X) = 1:size(X)[1] |> collect
rs(n, m) = X -> reshape(X, (n, m))
rbfT(x, y) = rbf(x, y, 1)


network = [
    constantl(10),
    rbfl(10, 10),
    constantl(30)
    # (10, @lf(rbf(_1, _2, 1.0) + poly(_1, _2, 1.0))),
    # constantl(10)
]
alpha, inds, S, cinds = repkmachine(network, X, Y)
begin
    nX = -5:.01:5 |> collect;
    plot(nX, repkpred(nX, alpha, inds, S), label = "RBF + Poly",  legend = :outertopright)
    plot!(nX, f.(nX), label = "Data")
end

begin
    inds = kmeansinit(X, Y, 10)
    K1 = ka(X, X[inds, :], @lf(rbf(_, _, 1.0)))
    K2 = ka(X, X[inds, :], @lf(poly(_, _, 1.0)))
    Xind = hcat(K1, K2)
    alpha = pinv(Xind)*Y
end

begin
    nX = -5:.01:5 |> collect;
    K1 = ka(nX, X[inds, :], @lf(rbf(_, _, 1.0)))
    K2 = ka(nX, X[inds, :], @lf(poly(_, _, 1.0)))
    nXind = hcat(K1, K2)
    plot(nX, nXind * alpha, label = "RBF cat Poly",  legend = :outertopright)
    plot!(nX, f.(nX), label = "Data")
end 