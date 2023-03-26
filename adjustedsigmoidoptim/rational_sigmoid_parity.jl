using Zygote, LinearAlgebra, Plots,Flux
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
using MLDatasets, MLUtils
train_x, train_y = MNIST.traindata(Float32)
test_x, test_y = MNIST.testdata(Float32)
TX = train_x |> flatten
tX = test_x |> flatten
toint(x) = floor(x) |> Int
rs(x, y) = w -> reshape(w, (x, y))

function viseval(i, m)
    [m(tX[:, i]); tX[:, i]] |> rs(28, 28*2) .|> Gray
end


sigmoid(x) = 1 * inv(1 + exp(-x))
rational_quadtratic(x) = 1 - inv(x^2 + 1)
gaussian_bell(x) = exp(-x^2/2)

rq = rational_quadtratic
s = sigmoid
gb = gaussian_bell

function create_model(n, w, acfunc)
    layers = [Dense(1, w); [Dense(w, w, acfunc) for _ in 1:n]; Dense(w, 1, acfunc)]
    return Chain(layers...)
end

let n = 1, w = 2, acfunc = gb, to_pred = s
    model = create_model(n, w, acfunc)
    p = Flux.params(model)
    opt = ADAM(.1)
    x = randn(100) .* 10 |> sort! |> rs(1, :)
    y = to_pred.(x)
    for epoch in 1:100
        gs = gradient(() -> Flux.mse(model(x), y), p)
        Flux.Optimise.update!(opt, Flux.params(model), gs)
    end
    plot(x[1, :], y[1, :], label = "target")
    plot!(x[1, :], model(x)[1, :], label = "model")
end

function trial(acfunc, lr, epochs)
    layers = [2, 1.4, 1.2, 1]
    w1 = (28 .^ layers[1:end-1]) .|> toint
    w2 = (28 .^ layers[2:end]) .|> toint
    encoder = Chain(
        Dense(w1[1], w2[1], acfunc),
        Dense(w1[2], w2[2], acfunc),
        Dense(w1[3], w2[3], acfunc),
    ) |> gpu
    decoder = Chain(
        Dense(w2[3], w1[3], acfunc),
        Dense(w2[2], w1[2], acfunc),
        Dense(w2[1], w1[1], acfunc),
    ) |> gpu
    model = Chain(encoder, decoder) |> gpu
    opt = ADAM(lr)
    data = Flux.DataLoader((TX, TX), batchsize = 312, shuffle = true)
    p = Flux.params(model)
    for epoch in 1:epochs
        println("Epoch: $epoch")
        for (x, y) in data
            let x = x |> gpu, y = y |> gpu
                gs = gradient(p) do
                    Flux.mse(model(x), y)
                end
                Flux.Optimise.update!(opt, Flux.params(model), gs)
            end
        end
    end
    Flux.mse(model(tX |> gpu), tX |> gpu) |> println
    model |> cpu
end
cpum =  trial(gb, .01, 10)
viseval(rand(1:1000), cpum)