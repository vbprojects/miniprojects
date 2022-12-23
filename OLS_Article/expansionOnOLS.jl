using Plots

normalpdf(x, μ, σ) = exp(-(x - μ)^2 / (2σ^2)) / (σ * sqrt(2π))
laplacianpdf(x, μ, b) = exp(-abs(x - μ) / b) / (2b * sqrt(2π))



x = -10:.1:10 |> collect
y = 2 .* x
z = zeros(length(x))
# plot(-10:.7:10, -20:1.4:20, (x, y) -> normalpdf(y, 2*x, 2), proj_type = :ortho, camera = (50,20), st=:wireframe)
plot3d(x, y, z, zlims = (0, .5))
plot!(-10:.1:10, -20:.1:20, (x, y) -> normalpdf(y, 2*x, 2), proj_type = :ortho, st=:surface, fillalpha=0.7, c = :roma)

plot3d(x, y, z, zlims = (0, .5))
plot!(-10:.1:10, -20:.1:20, (x, y) -> laplacianpdf(y, 2*x, 1), proj_type = :ortho, st=:surface, fillalpha=0.7, c = :roma)