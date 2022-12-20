using LinearAlgebra, MLDatasets,  Images, LambdaFn, Zygote

#import MNIST DATA
X, y = MNIST.traindata(Float32)

flatten(x) = reshape(x, :, size(x, 3))

ftrain = [X[:, :, x] |> flatten for x in 1:size(X, 3)]

ftrain[1] |> @lf(reshape(_', 28, 28)) .|> Gray

vis(X) = X |> @lf(reshape(_', 28, 28)) .|> Gray

Dense(input, output) = rand(input, output) .* .05

ftrain[1] + ftrain[2] |> vis

ftrain[1]' * Dense(28*28, 2000)

D1 = Dense(2000, 28*28)
D2 = Dense(2000, 2000)
D3 = Dense(2000, 2000)

(ftrain[1]' * D1) * D2 * D3

relu(x) = max.(x, 0)
sigmoid(x) = 1 / (1 + exp(-x))

function goodness(D, X)
    sigmoid(x) = 1 / (1 + exp(-x))
    return sigmoid(sum((D * X) .^ 2) - .5)
end

goodness(D1, ftrain[5])

neg = (ftrain[4] .+ ftrain[10]) ./ 2
neg |> vis

goodness(D1, neg)

(D1 * neg)

D1 = (rand(400, 28*28) .- .5) ./ (28*28)

function minmaxnorm(x)
    minval = minimum(x)
    maxval = maximum(x)
    return (x .- minval) ./ (maxval - minval)
end

relu(x) = max(0, x)
drelu(x) = x > 0 ? 1 : 0
grad(W, x) = 2 .* relu.(W * x) .* drelu.(W * x) * x'

grad(D1, ftrain[1]) |> minmaxnorm .|> Gray

begin
    D = (rand(400, 28*28) .- .5) ./ (28*28)
    for i in 1:length(y)
        G = grad(D, ftrain[i])
        if y[i] == 0 | y[i] == 1
            D = D .+ G .* .0001
        else
            D = D .- G .* .0001
        end
    end
end

