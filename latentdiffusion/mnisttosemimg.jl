using Zygote, LinearAlgebra, Flux, Statistics, Images, Random, CUDA, Plots
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
using MLDatasets, MLUtils
train_x, train_y = MNIST.traindata(Float32)
test_x, test_y = MNIST.testdata(Float32)
rs(x, y) = w -> reshape(w, (x, y))
data = DataLoader((TX, TX), batchsize=200, shuffle=true)
axis(x) = D -> [D[[k != x ? (:) : i for k in 1:length(size(D))]...] for i in axes(D, x)]

TX = train_x |> flatten
tX = test_x |> flatten
TY = train_y

toint(x) = floor(x) |> Int
layers = [2, 1.4, 1.2, 1]
w1 = (28 .^ layers[1:end-1]) .|> toint
w2 = (28 .^ layers[2:end]) .|> toint
encoder = Chain(
    Dense(w1[1], w2[1], selu),
    Dense(w1[2], w2[2], selu),
    Dense(w1[3], w2[3], sigmoid),
) |> gpu
decoder = Chain(
    Dense(w2[3], w1[3], selu),
    Dense(w2[2], w1[2], selu),
    Dense(w2[1], w1[1], sigmoid),
) |> gpu

loss = Flux.binarycrossentropy
opt = ADAM(0.01)
p = Flux.params(Chain(encoder, decoder))
for epoch in 1:20
    println("Epoch: $epoch")
    for (x_batch, y_batch) in data
        gpx = x_batch |> gpu
        gpy = y_batch |> gpu
        gs = gradient(p) do
            z = encoder(gpx)
            loss(decoder(z), gpy)
        end
        Flux.Optimise.update!(opt, p, gs)
    end
end

enc = encoder(TX |> gpu) |> cpu

encys = reduce(hcat, TY .|> y -> [(i != (y + 1)) ? 0 : 1 for i in 1:10]) .* 10


To_other_dist = Chain(Dense(38, 20, selu), Dense(20, 20, sigmoid))

To_other_dist([encys;enc])