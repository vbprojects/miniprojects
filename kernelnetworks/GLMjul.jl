using LinearAlgebra, LambdaFn, StatsBase, FLoops
using MacroTools

X = hcat(-2:.01:2, -2:.01:2) |> collect
Y = hcat(2 .* (-2:.01:2), 3 .* (-2:.01:2))
N = size(X)[1]
inds = [rand(1:N, 200) for _ in 1:4]
w = [pinv(X[inds[i], :]) * Y[inds[i], :] for i in 1:4]

function ka(X, Xinds, af)
    m = size(X)[1]
    n = size(Xinds)[1]
    K = zeros(m, n)
    @floop for i in 1:m
        for j in 1:n
            K[i, j] = af(X[i, :], Xinds[j, :])
        end
    end
    return K
end

function fkxi(X, Xinds, af)
    ka(X, Xinds, af), X, Xinds
end


function kxi(X, sam, n, af)
    inds = sam(n)
    K = ka(X, X[inds, :], af)
    K, X, (X[inds, :], af)
end

mac = :(begin 
    # R = []
    K, X, I = fkxi(X, X[rand(1:N, 2), :], @lf(_'*_ + 1))
    append!(R, I)
end)

rmlines(mac)


K, X = kaX(X, X[rand(1:N, 2), :], af)
K, X = kaX(K, K[rand(1:N, 2), :], af)



K, X, R = kxi(X, @lf(rand(1:N, _)), 2, af)
S = [R]
K, X, R = k(X, S[1][1], S[1][2])


K = ka(X, X[rand(1:N, 2), :], @lf(exp(-norm(_ - _)^2 / 2)))
Kl = ka(X, X[rand(1:N, 2), :], @lf(_'*_ + 1))
Kp = [ka(K, K[rand(1:N, 2), :], af) Kl]

w = [pinv(Kp[inds[i], :]) * Y[inds[i], :] for i in 1:4]

wavg = w |> @lf(mean(_, dims = 1))

Y .- (Kp * wavg[1]) |> @lf(sum(abs2, _))

[Y .- (Kp * w[i]) |> @lf(sum(abs2, _)) for i in 1:4]

