using LinearAlgebra, MLDatasets, Images, LambdaFn, CUDA, Plots

#import MNIST DATA
X, y = MNIST.traindata(Float32)

flatten(x) = reshape(x, :, size(x, 3))

ftrain = [X[:, :, x] |> flatten for x in 1:size(X, 3)]

ftrain[1] |> @lf(reshape(_', 28, 28)) .|> Gray

vis(X) = X |> @lf(reshape(_', 28, 28)) .|> Gray

relu(x) = max.(x, 0)
sigmoid(x) = 1 / (1 + exp(-x))

function goodness(D, X)
    sigmoid(x) = 1 / (1 + exp(-x))
    return sigmoid(sum((D * X) .^ 2) - .5)
end

neg = (ftrain[4] .+ ftrain[10]) ./ 2
neg |> vis

goodness(D1, neg)

function minmaxnorm(x)
    minval = minimum(x)
    maxval = maximum(x)
    return (x .- minval) ./ (maxval - minval)
end

relu(x) = max(0, x)
drelu(x) = x > 0 ? 1 : 0
grad(W, x) = 2 .* relu.(W * x) .* drelu.(W * x) * x'

grad(D1, ftrain[1]) |> minmaxnorm .|> Gray

D = (rand(400, 28*28) .- .5) ./ (28*28)

D * ftrain[1]

grad(D, ftrain[1])

i = 1
ftrain[3]

D1 = (rand(400, 28*28) .- .5) ./ (28*28)
D2 = (rand(2, 400) .- .5) ./ (28*28)
norma(x) = x ./ sum(x)
begin
    for _ in 1:20
        for i in 1:500
            G1 = grad(D1, ftrain[i])
            if y[i] == 0 || y[i] == 1 || y == 5
                D1 = D1 .+ G1 .* .0001
            else
                D1 = D1 .- G1 .* .0001 * 1/5
            end
            G2 = grad(D2, norma(D1 * ftrain[i]))
            if y[i] == 0 || y[i] == 1 || y == 5
                D2 = D2 .+ G2 .* .0001
            else
                D2 = D2 .- G2 .* .0001 * 1 / 5
            end
        end
    end
end
sinds = sortperm(y[1:500])
fsy = (y[sinds])
sy = fsy[1:116]
full = ftrain[sinds]
ftest = full[1:116]

sum(y[1:500] .== 0)
cla = [(D2 * norma(D1 * i)) for i in ftest]
sy

X = reduce(hcat, cla)
w = pinv(X)'*sy
sigmoid.(X'*w) .- sy |> @lf(sum(abs2, _))

X = reduce(hcat, ftrain[1:116])
w = pinv(X)'*sy
sigmoid.(X'*w) .- sy |> @lf(sum(abs2, _))

pnts = reduce(hcat, cla)'
scatter(pnts[:, 1], pnts[:, 2], group = sy)

fcla = [(D2 * norma(D1 * i)) for i in full]
pnts = reduce(hcat, fcla)'
scatter(pnts[:, 1], pnts[:, 2], group = fsy)

Dc = CuArray(D1)
img = CuArray.(ftrain[1:10])
D1 * reduce(hcat, ftrain[1:500])

