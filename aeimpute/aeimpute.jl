using Zygote, LinearAlgebra, Flux, Statistics, Images, Random, CUDA, Plots, Distributions, StatsAPI, Tables
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
using MLDatasets, MLUtils, DataFrames, MLJ, OpenML
axis(x) = D -> [D[[k != x ? (:) : i for k in 1:length(size(D))]...] for i in axes(D, x)]
# Read in OpenML dataset for benchmarking
begin
    data = OpenML.load(197)
end

data
# Normalize the data
begin
    M = data |> Tables.matrix
    M = M .- mean(M, dims=2)
    M = M ./ std(M, dims=2)
    M = M' |> collect
end

# Setup for training
epochs = 30
batch_size = 128
opt = ADAM(0.001)
loss = Flux.mse

shuffled = M[:, randperm(size(M, 2))]
train_X = shuffled[:, 1:Int(0.8 * size(M, 2) |> round)]
test_X = shuffled[:, Int(0.8 * size(M, 2) |> round):end]

w = size(train_X, 1)
N = size(train_X, 2)

model = Chain(Dense(w, w, selu), Dense(w, w, selu), Dense(w, w, selu), Dense(w, w, selu))
p = Flux.params(model)

begin
    losses = []
    for _ in 1:epochs
        missing_mat = rand(Bernoulli(0.5), size(train_X)) .|> Float32
        for i in Iterators.partition(randperm(N), batch_size)
            X = train_X[:, i]
            Xm = missing_mat[:, i]
            sXm = Xm .+ ((1 .- Xm) .* 1e-4)
            train_loss, back = Zygote.pullback(p) do
                loss(model(X .* (1 .- Xm)) .* sXm, X .* sXm)
            end
            grads = back(one(train_loss))
            push!(losses, train_loss)
            Flux.Optimise.update!(opt, p, grads)
        end
    end
end

bi_lin_model = Chain(Flux.Bilinear(w => w,selu), Dense(w, w, selu), Dense(w, w, selu))
bp = Flux.params(bi_lin_model)

begin
    blosses = []
    for _ in 1:epochs
        missing_mat = rand(Bernoulli(0.5), size(train_X)) .|> Float32
        for i in Iterators.partition(randperm(N), batch_size)
            X = train_X[:, i]
            Xm = missing_mat[:, i]
            sXm = Xm .+ ((1 .- Xm) .* 1e-4)
            train_loss, back = Zygote.pullback(bp) do
                loss(bi_lin_model((X .* (1 .- Xm), (1 .- Xm))) .* sXm, X .* sXm)
            end
            grads = back(one(train_loss))
            push!(blosses, train_loss)
            Flux.Optimise.update!(opt, bp, grads)
        end
    end
end

plot(losses, label = "Regular")
plot!(blosses, label = "Bilinear")

# Train loss
missing_mat = rand(Bernoulli(0.5), size(train_X)) .|> Float32
train_Xm = train_X .* (1 .- missing_mat)
loss(train_Xm .* missing_mat, bi_lin_model((train_Xm, (1 .- missing_mat))) .* missing_mat)
loss(train_Xm .* missing_mat, model(train_Xm) .* missing_mat)
# create missing mat for test set
missing_mat_test = rand(Bernoulli(0.1), size(test_X)) .|> Float32
test_Xm = test_X .* (1 .- missing_mat_test)

fm = loss(model(test_Xm) .* missing_mat_test, test_X .* missing_mat_test)
sm = loss(m2((test_Xm, missing_mat_test)) .* missing_mat_test, test_X .* missing_mat_test)

# mean impute
mean_impute = mean(test_Xm, dims=2)
missing_mat_test[1, :]

axis(1)(missing_mat_test)

imputed = [i * j[1] for (i, j) in zip(axis(1)(missing_mat_test), mean_impute)] |> x -> reduce(hcat, x)
mm = loss(imputed', test_X .* missing_mat_test)

fm, sm, mm