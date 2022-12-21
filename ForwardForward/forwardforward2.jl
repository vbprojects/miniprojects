using LinearAlgebra, MLDatasets,  Images, LambdaFn, Base.Iterators

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

function minmaxnorm(x)
    minval = minimum(x)
    maxval = maximum(x)
    return (x .- minval) ./ (maxval - minval)
end

t_y = (y .== 1 .|| y .== 0)
t_y = t_y .- (t_y .== 0) * sum(t_y) / length(t_y)

hstack(x) = reduce(hcat, x) |> collect

X = ftrain[1:32] |> hstack

D = (rand(400, 28*28) .- .5) ./ (28*28)

for _ in 1:10
    for p in partition(1:length(y[1:500]), 32)
        X = hstack(ftrain[p])
        pt_y = t_y[p]
        G = pgrad(D, X)
        D .-= sum(.0001 * (G[:, i] * pt_y[i]) * X[:, i]' for i in 1:length(p))
    end
end

X = relu.(D * hstack(ftrain[1:500]))
w = pinv(X)' * y[1:500]
pred = sigmoid.(w'*X)

pred