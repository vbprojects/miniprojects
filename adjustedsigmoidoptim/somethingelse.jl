using SymPy, Plots

x, y, w = symbols("x y w")

s = ((x*w)/sqrt(1+(x*w)^2)+1)/2

sl = (2*y - 1)*(y - s)^2

no = integrate(sl, w)

sl2 = (2*y - 1)*no

no2 = integrate(sl2, w)

sl3 = (2*y - 1)*no2

no3 = integrate(sl3, w)

sx = x/sqrt(1+x^2)
plot(sx)
plot!(integrate(sx))