using Zygote, LinearAlgebra, Flux, Statistics, Images, Random, CUDA, Plots, Distributions, StatsAPI
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
using MLDatasets, MLUtils
train_x, train_y = MNIST.traindata(Float32)
test_x, test_y = MNIST.testdata(Float32)
TX = train_x |> flatten
tX = test_x |> flatten
rs(x, y) = w -> reshape(w, (x, y))
data = DataLoader((TX, TX), batchsize=200, shuffle=true)
axis(x) = D -> [D[[k != x ? (:) : i for k in 1:length(size(D))]...] for i in axes(D, x)]


toint(x) = floor(x) |> Int
layers = [2, 1.4, 1.2, 1]
w1 = (28 .^ layers[1:end-1]) .|> toint
w2 = (28 .^ layers[2:end]) .|> toint
binary_step(x) = (x .> 0.5)
encoder = Chain(
    Conv((3, 3), 1 => 16, selu, pad=(1, 1)),
    Conv((3, 3), 16 => 32, selu, pad=(1, 1)),
    Conv((3, 3), 32 => 64, selu, pad=(1, 1)),
    binary_step
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

encdec = Chain(encoder, decoder) |> cpu
Flux.kldivergence(encdec(tX), tX)

iS = rand(1:1000, 4)
[reduce(hcat, [encdec(TX[:, i])' |> rs(28, 28) for i in iS]); reduce(hcat, [TX[:, i]' |> rs(28, 28) for i in iS])] .|> Gray

enc = encoder |> cpu
dec = decoder |> cpu
enc(TX)'
oneat(x) = begin
    v = zeros(28)
    v[x] = 1
    v
end

X = reduce(hcat, [oneat(i) for i in 1:5])

d = Distributions.Product([Bernoulli(.5) for _ in 1:28])

rand(d) |> dec |> rs(28, 28) |> transpose .|> Gray

Chain(
    Conv((3, 3), 1 => 16, selu, pad=(1, 1)),
    Conv((3, 3), 16 => 32, selu, pad=(1, 1)),
    Conv((3, 3), 32 => 64, selu, pad=(1, 1)),
    
    )