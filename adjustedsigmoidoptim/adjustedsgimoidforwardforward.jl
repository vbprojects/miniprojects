using LinearAlgebra, MLDatasets,  Images, LambdaFn, Base.Iterators, Zygote, Plots, Statistics

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

function objd(w, I, y)
    yh = sum(abs2, relu.(w * I))
    Lwrtw = gradient(w -> (relu.(w * I)' * relu.(w * I))[1], w)[1]
    abs(y - yh) * (2*y - 1) * Lwrtw
end

function minmaxnorm(x)
    minval = minimum(x)
    maxval = maximum(x)
    return (x .- minval) ./ (maxval - minval)
end


t_y = (y .== 9 .|| y .== 5)
p_i = findall(==(1), t_y)
n_i = findall(==(0), t_y)

w = rand(200, 28*28) .- .5

for _ in 1:500
    pn = rand(0:1)
    if pn == 1
        I = ftrain[rand(p_i)]
    else
        I = ftrain[rand(n_i)]
    end
    w .+= .0001 * objd(w, I, pn)
end

w |> minmaxnorm .|> Gray


syh(w, I) = sigmoid(sum(abs2, w * I))

ftrain[p_i[1]] |> vis

p_i

n_i

mean([syh(w, ftrain[i]) for i in p_i])

mean([syh(w, ftrain[i]) for i in n_i[1:length(p_i)]])

syh(w, ftrain[5])

sum(abs2, relu.(w * ftrain[5]))

p_i



