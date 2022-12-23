using LinearAlgebra, MLDatasets,  Images, LambdaFn, Base.Iterators, Zygote, Plots

#import MNIST DATA
X, y = MNIST.traindata(Float32)

flat(x) = reshape(x, :, size(x, 3))

ftrain = [X[:, :, x] |> flat for x in 1:size(X, 3)]

ftrain[1] |> @lf(reshape(_', 28, 28)) .|> Gray

vis(X) = X |> @lf(reshape(_', 28, 28)) .|> Gray

relu(x) = max(0, x)
drelu(x) = x > 0 ? 1 : 0
grad(W, x) = 2 .* relu.(W * x) .* drelu.(W * x) * x'
pgrad(W, x) = 2 .* relu.(W * x) .* drelu.(W * x)
sigmoid(x) = 1 / (1 + exp(-x))
gilu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*x + 0.044715*x^3))
normalize(x) = x ./ sum(x)

function minmaxnorm(x)
    minval = minimum(x)
    maxval = maximum(x)
    return (x .- minval) ./ (maxval - minval)
end

t_y = (y .== 1 .|| y .== 5)

hstack(x) = reduce(hcat, x) |> collect
N = 5000
D = (rand(200, 28*28) .- .5) ./ (28*28)
oD = deepcopy(D)

D2 = (rand(2, 200) .- .5) ./ (28*28)

function obj(X, Y, D)
    K = relu.(D * X)
    pred = sigmoid.(diag(K' * K) .- .5)
    dif = pred .- Y
    dif' * dif
end

for _ in 1:100
    for p in partition(1:length(y[1:N]), 32)
        X = hstack(ftrain[p])
        Y = t_y[p]

        G = gradient(D -> obj(X, Y, D), D)[1]
        D .-= .0001 * G

        X2 = relu.(D * X)

        G2 = gradient(D -> obj(X2, Y, D), D2)[1]
        D2 .-= .0001 * G2
    end
end

D |> gvis
oD |> gvis

(D .- oD) .^ 2 |> gvis

Y = t_y[1:N]
X = hstack(ftrain[1:N])
K = relu.(D * X)
w = pinv(K)' * Y
pred = (w'*K .- .5) .|> sigmoid |> @lf(_.>.5)
(Y' .- pred) .^ 2 |> mean

pnts = relu.(D2 * relu.(D * X))

scatter(pnts[1, :], pnts[2, :], group=y[1:N]) 