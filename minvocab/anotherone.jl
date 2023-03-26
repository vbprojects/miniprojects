using Zygote, LinearAlgebra, Flux, Statistics, Images, Random, CUDA, Plots
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
using MLDatasets, MLUtils
train_x, train_y = MNIST.traindata(Float32)
test_x, test_y = MNIST.testdata(Float32)
axis(x) = D -> [D[[k != x ? (:) : i for k in 1:length(size(D))]...] for i in axes(D, x)]
TX = train_x |> flatten
tX = test_x |> flatten
rs(x, y) = z -> reshape(z, (x, y))
data = DataLoader((TX, TX), batchsize=200, shuffle=true)

begin
    encoder = Chain(
        Dense(784, 500, selu),
        Dense(500, 100, selu),
        Dense(100, 78, selu),
        x -> reshape(x, (3, 26, :)),
    )

    middle = Chain(
        x -> softmax(x, dims = 2)
    )

    decoder = Chain(
        flatten,
        Dense(78, 100, selu),
        Dense(100, 500, selu),
        Dense(500, 784, selu)
    )
end
full = Chain(encoder, middle, decoder) |> gpu

loss = Flux.mse

opt = ADAM(0.01)
p = Flux.params(full)
for epoch in 1:20
    println("Epoch: $epoch")
    for (x_batch, y_batch) in data
        gpx = x_batch |> gpu
        gpy = y_batch |> gpu
        gs = gradient(p) do
            loss(full(gpx), gpy)
        end
        Flux.Optimise.update!(opt, p, gs)
    end
end

Flux.mse(full(tX |> gpu), tX |> gpu)

cpum = full |> cpu
letters = Chain(encoder, middle) |> cpu

alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

tX[:, findall(==(2), train_y)[10]] |> letters |> axis(1) .|> argmax |> x -> alphabet[x]