using Zygote, LinearAlgebra, Plots, Flux, MLJ
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
using MLDatasets, MLUtils
rational_quadtratic(x) = (x .^ 2) ./ ((x .^ 2) .+ 1)
sigmoid(x) = @. 1 / (1 + exp(-x))
tocolor(x) = x .> 0.5 ? :red : :blue
X, y = make_moons(1000, noise=0.2, as_table = false)
scatter(X[:, 1], X[:, 2], color = y)

rational_cubic(x) = @. x^3 / (x*(x^2 + 1))

begin
    acfunc = rational_quadtratic
    layers = 4
    model = Chain([[Dense(2, 2, acfunc) for _ in 1:layers]; Dense(2, 1, acfunc)]...)
    loss(x, y) = Flux.mse(model(x), y)
    opt = ADAM(0.01)
    data = Flux.DataLoader((X', reshape(y, (1, :))), batchsize = 32, shuffle = true)
    p = Flux.params(model)
    for epoch in 1:100
        for (x, y) in data
            gs = gradient(() -> loss(x, y), p)
            Flux.Optimise.update!(opt, Flux.params(model), gs)
        end
    end
    scatter(X[:, 1], X[:, 2], color = (model(X') |> x -> x[1, :] .> 0.5) .|> tocolor)
end



train_x, train_y = MNIST.traindata(Float32)
test_x, test_y = MNIST.testdata(Float32)
TX = train_x |> flatten
tX = test_x |> flatten
rs(x, y) = w -> reshape(w, (x, y))
data = DataLoader((TX, TX), batchsize=200, shuffle=true)
axis(x) = D -> [D[[k != x ? (:) : i for k in 1:length(size(D))]...] for i in axes(D, x)]


rational_quadtratic(x) = @. 1/(1 + x^2)
rq(x) = 1 - inv(1 + ^(x, 2))
gpdf(x) = exp(-^(x, 2)/2)
acfunc = gpdf
encoder = Chain([Dense(784, 314, acfunc), Dense(314, 112, acfunc), Dense(112, 28, acfunc)]...) |> gpu
decoder = Chain(reverse([Dense(314, 784, acfunc), Dense(112, 314, acfunc), Dense(28, 112, acfunc)])...) |> gpu
model = Chain(encoder, decoder) |> gpu
opt = ADAM(0.005)
data = Flux.DataLoader((TX, TX), batchsize = 312, shuffle = true)
p = Flux.params(model)
for epoch in 1:10
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
Flux.mse(model(tX |> gpu), tX |> gpu)
cpum = model |> cpu

function viseval(i, m)
    [m(tX[:, i]); tX[:, i]] |> rs(28, 28*2) .|> Gray
end

viseval(19, cpum)