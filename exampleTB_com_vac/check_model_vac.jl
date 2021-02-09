using JuLIP
using ACE
using IPFitting
using Plots
using LinearAlgebra
using MDLearn
using ASE

r0 = rnn(:Ti)
    
Bpair = pair_basis(species = :Ti,
    r0 = r0,
    maxdeg = 3,
    rcut = 7.0,
    pcut = 1,
    pin = 0)

Bsite = rpi_basis(species = :Ti,
      N = 3,                       # correlation order = body-order - 1
      maxdeg = 16, #10           # polynomial degree
      r0 = r0,                      # estimate for NN distance
      #D = SparsePSHDegree(; wL=1.3, csp=1.0),
      rin = 2.2362368236146413, rcut = 5.5,   # domain for radial basis (cf documentation) #5.5
      pin = 2);

weights = Dict(
    "ignore"=> [],
    "default" => Dict("E" => 10.0, "F" => 1.0 , "V" => 1.0 ),
    )

Vref = OneBody(:Ti => -5.817622899211898)

B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

for i in 11:11
    al = IPFitting.Data.read_xyz(@__DIR__() * "/DB_$(i-1).xyz", energy_key="energy", force_key="force");

    dB = LsqDB("", B, al);

    IP, lsqinfo = lsqfit(dB, solver=(:rrqr, 1e-8), weights = weights, Vref = Vref, asmerrs=true);

    norm(lsqinfo["c"])
    rmse_table(lsqinfo["errors"])   

    save_dict(@__DIR__() * "/Ti_NRLTB_vac_3_N3_16_DB$(i).json", Dict("IP" => write_dict(IP), "info" => lsqinfo))
end






#at = al[1].at
# temp=6000

# nsteps = 10000
# E_tot = zeros(nsteps)
# E_pot = zeros(nsteps)
# E_kin = zeros(nsteps)
# T = zeros(nsteps)
# P = zeros(nsteps)
# varEs = zeros(nsteps)
# varFs = zeros(nsteps)

# E0 = energy(IP, at)

# at = MDLearn.MD.MaxwellBoltzmann_scale(at, temp)
# at = MDLearn.MD.Stationary(at)

# cfgs = []
# dt = 1.0

# for i in 1:nsteps
#     @show i
#     at = MDLearn.MD.VelocityVerlet(IP, at, dt * MDLearn.MD.fs)
#     Ek = ((0.5 * sum(at.M) * norm(at.P ./ at.M)^2)/length(at.M)) / length(at.M)
#     Ep = (energy(IP, at) - E0) / length(at.M)
#     E_tot[i] = Ek + Ep
#     E_pot[i] = Ep
#     E_kin[i] = Ek
#     T[i] = Ek / (1.5 * MDLearn.MD.kB)
#     if i % 10 == 0
#         push!(cfgs, ASEAtoms(at))
#     end
# end

# length(cfgs)

# for i in 1:100:1000
#     write_xyz("./exampleTB_com/img_$(i).xyz", cfgs[i])
# end

# plot(E_tot)
# plot!(E_kin)
# plot!(E_pot)



# Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
#                             Vref=Vref, Ibasis = :,Itrain = :,
#                             weights=weights, regularisers = [])

# m, S = MDLearn.Uncertain.posterior(Ψ, Y, 5.0, 5.0);
# c_samples = MDLearn.Uncertain.do_brr(Ψ, Y, 5.0, 5.0, 20);
# #IP, lsqinfo = lsqfit(dB, weights = weights, Vref = Vref, asmerrs=true);
# IP = JuLIP.MLIPs.SumIP(Vref, JuLIP.MLIPs.combine(B, c_samples[:,1]))