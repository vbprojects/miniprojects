using Zygote, LinearAlgebra, MLJ, Plots

gr()

sig(w, x) = 1 / (1 + exp(-w' * x))
loss(w, x, y) = w' * ((2*y - 1)*(y - 1)^2*x) - (2*y - 1)*sig(w, x) + (2*y - 1)^2 * log(sig(w, x))
dloss(w, x, y) = (2*y - 1)*(y - sig(w, x))^2*x
sloss(w, x, y) = -y*log(sig(w, x)) - (1 - y)*log(1 - sig(w, x))
zdloss(w, x, y) = Zygote.gradient(w -> loss(w, x, y), w)[1]
hloss(w, x, y) = max(0, 1 - (2*y - 1)*(w'* x))
X, Y = make_blobs(100,1, centers = 2, as_table = false)
X = [X ones(size(X, 1))]
Y = Y .- 1
w = zeros(2)
k = 10
dloss(w, X[k, :], Y[k]), zdloss(w, X[k, :], Y[k])
loss(w, X[k, :], Y[k])


Loss(w, X, Y) = sum(loss(w, X[i, :], Y[i]) for i in 1:size(X)[1])
SLoss(w, X, Y) = sum(sloss(w, X[i, :], Y[i]) for i in 1:size(X)[1])
HLoss(w, X, Y) = sum(hloss(w, X[i, :], Y[i]) for i in 1:size(X)[1])



I = -20:.5:20
K = -20:.5:20
z = [Loss([i;k], X, Y) for i in I, k in K]
zs = [SLoss([i;k], X, Y) for i in I, k in K]
zh = [HLoss([i;k], X, Y) for i in I, k in K]
plotlyjs()
surface(I, K, abs.(z .- zs))
surface(I, K, z, color = :blues, legend = false)
surface!(I, K, zs, color = :reds, legend = false)
surface!(I, K, zh, color = :greens, legend = false)
gr()

z .- zs
reduce(+, [zdloss(w, X[i, :], Y[i]) for i in 1:size(X)[1]])