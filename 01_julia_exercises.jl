
####################################################################
# please mind Julia and libraris' versions:
#
# JULIA_VERSION=1.2.0
# Pkg.add(Pkg.PackageSpec(;name="Turing", version="0.6.23")
# Pkg.add(Pkg.PackageSpec(;name="Plots", version="0.25.3"))
# Pkg.add(Pkg.PackageSpec(;name="Distributions", version="0.21.1"))
# Pkg.add(Pkg.PackageSpec(;name="StatsBase", version="0.32.0"))
####################################################################


println("All models are wrong, but some are useful.")

##############################################
# basic Julia syntaxix
##############################################

# write you command here


# substract here


println("computed ", 3 - 4.5) 

println("Hello" * " Julia!")  

" who" ^ 3 * ", who let the dogs out" # do you know this song?

println((50 / 60) ^ 2)

x = 9
y = 7

# divide here


rem(9, 7)

# add x and y here


# divide y over x without using 'x / y'

x / y

false      &&   true       ||  true 

x = 3

# 4x^2+x



4+1<2||2+2<1

a=true
b=!!!!a

θ = 3

2θ

typeof(1)

typeof("Hello, world!")

typeof("αβγδ")     

typeof("H")

typeof('H')

typeof(1+3+5.)

Array{Int64}(undef, 3)

Array{String}(undef, 3)

'a'==="a"

x=3
y=2
a = Array{Integer,2}(undef, x, y)

x = Array{Int64}(undef,11, 12) 
typeof(x)

a, b, c = cos(0.2), log(10), abs(-1.22)  # multiple assignment

myfunc(x) = 20*x

myfunc(2)

add(x,y) = x + y

add(33, -22.2) 

function nextfunc(a, b, c) 
    a*b + c                
end

nextfunc(7,5,3)  

function print_type(x)
    println("The type of testvar is $(typeof(x)) and the value of testvar is $x")
end

a = ['1',2.]
print_type(a)

function f(x)
  return 2x
  3x
end

f(5)

cos_func(x) = cos(x)

cos_func(.7) 

cos_func(adj, hyp) = adj/hyp

cos_func(.7) 

cos_func(12, 13)

methods(cos_func)

?cos_func

cos_func(theta::Float64) = cos(theta)   # :: forces Julia to check the type

?cos

x = [1, 6, 2, 4, 7, 2, 76, 5]

# size(x)



#length(x)



typeof(x)

x = collect(1:7)

x = range(0, stop = 1, length = 10)

x = collect(x)

x = fill(5, 4)

x + 2

#x .+ 2


x .* x

data = [1.6800483  -1.641695388; 
        0.501309281 -0.977697538; 
        1.528012113 0.52771122;
        1.70012253 1.711524991; 
        1.992493625 1.891000015]

# find the size of data


typeof(data)

data[:,1] # all rows, 1st column

data = [[3,2,1] [3,2,1] [3,2,1] [3,2,1] [3,2,1] [3,2,1] [3,2,1] [3,2,1] [6,5,4]]

data[2,:]

rows, cols = size(data)
println(rows)
println(cols)

for num = 1:2:length(x)
    println("num is now $num")
end

values = [ "first element", 'θ' , 42]      # an array

for x in values    
    println("The value of x is now $x")
end

x = [3, 2, 1]

count=1
for i in x
  x[i]=count
  count=count+1
end

println(x[3])
println(x)

if 5>3
    println("test passed")
end

using Plots

gr()

x = cos.( - 10:0.1:10)
plot(1:length(x), x)

plot!(title = "First plot", size = [300, 300], legend = false) # modify existing plot

hline!([0])

plot(1:length(x), x, 
    size = [300, 300], 
    legend=false, 
    line= :scatter,
    marker = :hex)

x = collect(1:0.1:7)
f(x) = 2 - 2x + x^2/4 + sin(2x)
plot(x, f)

#savefig("Plot_name.png")                           # Saved png format

using Distributions

d = Normal()

n_draws = 1000
draws = rand(d, n_draws)
histogram(draws, size = [300, 300], bins = 40, leg = false, norm = true)

[pdf(d, x) for x in -3:1:3] # evaluate PDF/PMF

shape = (3, 2)
rand(Uniform(0,1), shape)
