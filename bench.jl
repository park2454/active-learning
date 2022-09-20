include("functions.jl")
using GLM, DataFrames, Statistics, DelimitedFiles, Random, Dates, LinearAlgebra, BenchmarkTools, JLD2, FileIO
import StatsBase.sample

magic = readdlm("magic04.data", ',', Any, '\n')
loc = findall(x->x=="g",magic[:,11])
magic[loc,11] .= 1
loc = findall(x->x=="h",magic[:,11])
magic[loc,11] .= 0
magic = convert(Array{Float64}, magic)
magic = [ones(size(magic)[1]) magic]

Y = magic[:,12]
X = magic[:,1:11]
n, p = size(X)

Ncand = 100:100:1000
Np = 2:11
#N2 = 100:100:1000
#Np = 1:11


#=
samp1 = sample(1:n, N1, replace = false)
cand = sample(setdiff(1:n,samp1), N2, replace = false)
p_var = [1; sample(2:p, Np-1, replace = false)]
data = DataFrame(magic[samp1, [p_var;12]])

formula_array = [
    @formula(x2~x1+0), 
    @formula(x3~x2+x1+0), 
    @formula(x4~x3+x2+x1+0),
    @formula(x5~x4+x3+x2+x1+0),
    @formula(x6~x5+x4+x3+x2+x1+0),
    @formula(x7~x6+x5+x4+x3+x2+x1+0), 
    @formula(x8~x7+x6+x5+x4+x3+x2+x1+0),
    @formula(x9~x8+x7+x6+x5+x4+x3+x2+x1+0),
    @formula(x10~x9+x8+x7+x6+x5+x4+x3+x2+x1+0),
    @formula(x11~x10+x9+x8+x7+x6+x5+x4+x3+x2+x1+0),
    @formula(x12~x11+x10+x9+x8+x7+x6+x5+x4+x3+x2+x1+0)
    ]
β = coef(glm(formula_array[Np], data, Binomial(), LogitLink()))
p = logistic(X*β)
w = p .* (1 .- p)

X_cand = Diagonal(sqrt.(w[cand]))*X[cand,p_var]
X_old = Diagonal(sqrt.(w[samp1]))*X[samp1,p_var]
=#

Ncand = 100:100:1000
Np = 2:11

SOCP_memory = zeros(Int64, 10,10)
SOCP_sample = zeros(Int64, 10,10)
SOCP_median = zeros(Float64, 10,10)

MISOCP_memory = zeros(Int64, 10,10)
MISOCP_sample = zeros(Int64, 10,10)
MISOCP_median = zeros(Float64, 10,10)

for i = 1:10
    for j = 1:i
        cand = sample(1:n, Ncand[i+1-j], replace=false)
        p_var = [1;sample(1:p, Np[j]-1,replace=false)]
        X_cand = X[cand,p_var]
        X_old = zeros(1,length(p_var))
        half = Int(Ncand[i+1-j]/2)
        bmk = @benchmark sagnol_A($X_cand, $X_old, $half;K = ($X_cand)', verbose=0, IC=0) seconds=7200 samples=10 evals=1
        SOCP_memory[j,i+1-j] = bmk.memory
        SOCP_sample[j,i+1-j] = length(bmk.times)
        SOCP_median[j,i+1-j] = median(bmk.times) * 10^-9
        @save "bench.jld2" SOCP_memory SOCP_sample SOCP_median MISOCP_memory MISOCP_sample MISOCP_median
        println([j;i+1-j])
        println("SOCP")
    end
end

for i = 1:10
    for j = 1:i
        cand = sample(1:n, Ncand[i+1-j], replace=false)
        p_var = [1;sample(1:p, Np[j]-1,replace=false)]
        X_cand = X[cand,p_var]
        X_old = zeros(1,length(p_var))
        half = Int(Ncand[i+1-j]/2)
        bmk = @benchmark sagnol_A($X_cand, $X_old, $half;K = ($X_cand)', verbose=0, IC=1) seconds=7200 samples=10 evals=1
        MISOCP_memory[j,i+1-j] = bmk.memory
        MISOCP_sample[j,i+1-j] = length(bmk.times)
        MISOCP_median[j,i+1-j] = median(bmk.times) * 10^-9
        @save "bench.jld2" SOCP_memory SOCP_sample SOCP_median MISOCP_memory MISOCP_sample MISOCP_median
        println([j;i+1-j])
        println("MISOCP")
    end
end

for i = 11:19
    for j = (i-9):10
        cand = sample(1:n, Ncand[i+1-j], replace=false)
        p_var = [1;sample(1:p, Np[j]-1,replace=false)]
        X_cand = X[cand,p_var]
        X_old = zeros(1,length(p_var))
        half = Int(Ncand[i+1-j]/2)
        bmk = @benchmark sagnol_A($X_cand, $X_old, $half;K = ($X_cand)', verbose=0, IC=0) seconds=7200 samples=10 evals=1
        SOCP_memory[j,i+1-j] = bmk.memory
        SOCP_sample[j,i+1-j] = length(bmk.times)
        SOCP_median[j,i+1-j] = median(bmk.times) * 10^-9
        @save "bench.jld2" SOCP_memory SOCP_sample SOCP_median MISOCP_memory MISOCP_sample MISOCP_median
        println([j;i+1-j])
        println("SOCP")
    end
end

for i = 11:19
    for j = (i-9):10
        cand = sample(1:n, Ncand[i+1-j], replace=false)
        p_var = [1;sample(1:p, Np[j]-1,replace=false)]
        X_cand = X[cand,p_var]
        X_old = zeros(1,length(p_var))
        half = Int(Ncand[i+1-j]/2)
        bmk = @benchmark sagnol_A($X_cand, $X_old, $half;K = ($X_cand)', verbose=0, IC=1) seconds=7200 samples=10 evals=1
        MISOCP_memory[j,i+1-j] = bmk.memory
        MISOCP_sample[j,i+1-j] = length(bmk.times)
        MISOCP_median[j,i+1-j] = median(bmk.times) * 10^-9
        @save "bench.jld2" SOCP_memory SOCP_sample SOCP_median MISOCP_memory MISOCP_sample MISOCP_median
        println([j;i+1-j])
        println("MISOCP")
    end
end