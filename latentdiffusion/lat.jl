using Zygote, LinearAlgebra, Flux, Statistics, Images, Random, CUDA, Plots
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
using MLDatasets, MLUtils
train_x, train_y = MNIST.traindata(Float32)
test_x, test_y = MNIST.testdata(Float32)

TX = train_x |> flatten
tX = test_x |> flatten

data = DataLoader((TX, TX), batchsize=200, shuffle=true)

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

loss = Flux.mse
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

begin
    iS = rand(1:1000, 4)
    [reduce(hcat, [encdec(TX[:, i])' |> rs(28, 28) for i in iS]); reduce(hcat, [TX[:, i]' |> rs(28, 28) for i in iS])] .|> Gray
end

data = DataLoader((TX, TX), batchsize=625, shuffle=true)
loss = Flux.mse
opt = Adam(0.01)
diffuser = Chain(Dense(28, 28, selu), Dense(28, 28, selu), Dense(28, 28, sigmoid)) |> gpu
p = Flux.params(diffuser)
for epoch in 1:20
    println("Epoch: $epoch")
    for (x_batch, y_batch) in data
        gpx = x_batch |> gpu
        gpy = y_batch |> gpu
        z = encoder(gpx)
        S = size(z, 2)
        nz = z .+ rand() .* (z[:, rand(1:S, S)] .- z[:, rand(1:S, S)])
        gs = gradient(p) do
            loss(decoder(diffuser(nz)), gpy)
        end
        Flux.Optimise.update!(opt, p, gs)
    end
end

rs(x, y) = w -> reshape(w, (x, y))

fm = Chain(encoder, diffuser, decoder) |> cpu

Flux.kldivergence(fm(tX), tX)

begin
    iS = rand(1:1000, 4)
    [reduce(hcat, [fm(TX[:, i])' |> rs(28, 28) for i in iS]); reduce(hcat, [TX[:, i]' |> rs(28, 28) for i in iS])] .|> Gray
end


fm(flatten(train_x)[:, rand(1:1000)]) |> rs(28, 28) .|> Gray
encdec(TX[:, rand(1:1000)])' |> rs(28, 28) .|> Gray
X = test_x |> flatten
dd(randn(28)) |> rs(28, 28) .|> Gray



X = test_x |> flatten
proc1() = begin    
    L = axes(X, 2)
    r1 = X[:, rand(L)]
    r2 = X[:, rand(L)]
    r3 = X[:, rand(L)]
    s = rand() - 2
    I1 = r1 .+ s .* (r2 .- r3) |> fm |> rs(28, 28)
    I2 = r1 .+ s .* (r2 .- r3) |> encdec |> rs(28, 28)
    I3 = r1 |> rs(28, 28)
    I4 = r2 |> rs(28, 28)
    I5 = r3 |> rs(28, 28)
    I6 = abs.(I1 .- I2)
    [I1' I2' I6' ; I3' I4' I5'] .|> Gray 
end

dd = Chain(diffuser, decoder) |> cpu
d = Chain(decoder) |> cpu
proc2() = begin
    z = rand(28)
    I1 = z |> dd |> rs(28, 28)
    I2 = z |> d |> rs(28, 28)
    [I1' I2'] .|> Gray
end

tX = test_x |> flatten

Flux.kldivergence(d(rand(28, 60000)), TX)
Flux.kldivergence(dd(rand(28, 60000)), TX)

enc = encoder |> cpu

function sample(enc, n)
    L = axes(X, 2)
    r1 = X[:, rand(L, n)] |> enc
    r2 = X[:, rand(L, n)] |> enc
    r3 = X[:, rand(L, n)] |> enc
    s = rand(n)
    r1 .- (r2 .- r3) .* s'
end

sampled = sample(enc, 60000)

NX = dd(sampled)

TY = train_y |> flatten
FNY = [ones(size(TY, 2));zeros(size(TY, 2))]
FNX = [TX NX]

data = DataLoader((FNX, FNY), batchsize=1000, shuffle=true)

forwardforward = Chain(Dense(w1[1], 2000, selu)) |> gpu
opt = ADAM(0.01)
p = Flux.params(forwardforward)
loss = Flux.binarycrossentropy

for epoch in 1:20
    println("Epoch: $epoch")
    for (x_batch, y_batch) in data
        gpx = x_batch |> gpu
        gpy = y_batch |> gpu
        gs = gradient(p) do
            Yp = sigmoid(sum(forwardforward(gpx) .^ 2, dims = 1) .- 500)
            loss(gpy, Yp')
        end
        Flux.Optimise.update!(opt, p, gs)
    end
end

forwardforward(NX|> gpu)