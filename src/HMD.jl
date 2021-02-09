module HMD

using JuLIP, Distributions, LinearAlgebra
using JuLIP.MLIPs: SumIP

export VelocityVerlet_uncertain, VelocityVerlet_uncertain2, get_energy_forces, uncertain_forces

function VelocityVerlet_uncertain(IP, B, at, dt, S, c; τ = 0.1)
    V = at.P ./ at.M
    F1 = forces(IP, at)
    F1σ = uncertain_forces(B, at, S, c)
    F1 = F1 + τ*F1σ

    rMSF1 = norm(vcat(F1...))
    rMSF1σ = norm(vcat(τ*F1σ...))

    p = rMSF1σ/rMSF1

    A = F1 ./ at.M

    set_positions!(at, at.X + (V .* dt) + (.5 * A * dt^2))

    F2 = forces(IP, at) + τ*uncertain_forces(B, at, S, c)
    nA = F2 ./ at.M
    nV = V + (.5 * (A + nA) * dt)
    set_momenta!(at, nV .* at.M)

    return at, p
end

function uncertain_forces(B, at, S, c)
    return sum(c .* ((S) * forces(B, at)) )
end

################################

function VelocityVerlet_uncertain_com(Vref, B, c_samples, at, dt; τ = 1e-10)
    V = at.P ./ at.M
    meanE, varE, meanF, varF = get_energy_forces(Vref, B, c_samples, at);    
    F1 = meanF + τ*varF
    
    A = F1 ./ at.M

    set_positions!(at, at.X + (V .* dt) + (.5 * A * dt^2))
    meanE, varE, meanF, varF = get_energy_forces(Vref, B, c_samples, at)
    F2 = meanF + τ*varF
    nA = F2 ./ at.M
    nV = V + (.5 * (A + nA) * dt)
    set_momenta!(at, nV .* at.M)

    rMSF1 = norm(vcat(meanF...))
    rMSF1σ = norm(vcat(τ*varF...))

    p = rMSF1σ/rMSF1

    return at, p
end

function get_energy_forces(Vref, B, c_samples, at)
    E_shift = energy(Vref, at)

    nIPs = length(c_samples[1,:])

    E = energy(B, at)
    F = forces(B, at)
    
    Es = [E_shift + sum(c_samples[:,i] .* E) for i in 1:nIPs];
    Fs = [sum(c_samples[:,i] .* F) for i in 1:nIPs];
    
    meanE = mean(Es)
    varE = sum([ (Es[i] - meanE)^2 for i in 1:nIPs])/nIPs
    
    meanF = mean(Fs)
    varF =  sum([ 2*(Es[i] - meanE)*(Fs[i] - meanF) for i in 1:nIPs])/nIPs
    
    return meanE, varE, meanF, varF
end


end