module repkern
mutable struct repkern
    Xinds # Dict of matrices at inds
    inds # Dict of indices
    S # Dict of functions
    alpha # Vector of weights
    layers # Dict of layers
end
end

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


