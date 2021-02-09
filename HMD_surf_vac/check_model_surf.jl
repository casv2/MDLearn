using JuLIP
using ACE
using IPFitting
using LinearAlgebra

r0 = rnn(:Ti)

al = IPFitting.Data.read_xyz(@__DIR__() * "/DB_30.xyz", energy_key="energy", force_key="force", virial_key="virial")
@show length(al)
al = filter(at -> configtype(at) != "FLD_hcp", al)
al = filter(at -> configtype(at) != "FLD_bcc", al)

EV = IPFitting.Data.read_xyz("./ev_ph/FLD16_hcp_bcc_sm2-6+3p.xyz", energy_key="energy", force_key="force", virial_key="virial")
EV2 = IPFitting.Data.read_xyz("./ev_ph/FLD16_hcp_bcc_sm3-6+3p.xyz", energy_key="energy", force_key="force", virial_key="virial")
al2 = vcat(al, EV, EV2)

R = minimum(IPFitting.Aux.rdf(al2, 5.0));
0.8*rnn(:Ti)


Bpair = pair_basis(species = :Ti,
    r0 = r0,
    maxdeg = 12,
    rcut = 7.0,
    pcut = 1,
    pin = 0)

Bsite = rpi_basis(species = :Ti,
      N = 4,                       # correlation order = body-order - 1
      maxdeg = 18, #10           # polynomial degree
      r0 = r0,                      # estimate for NN distance
      #D = SparsePSHDegree(; wL=1.3, csp=1.0),
      rin = R, rcut = 5.5,   # domain for radial basis (cf documentation) #5.5
      pin = 2);

B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

@show length(B)

dB = LsqDB("", B, al2);

weights = Dict(
    "ignore"=> [],
    "default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ),
    "PH_bcc" => Dict("E" => 15.0, "F" => 100.0 , "V" => 1.0 ),
    #"HMD_bcc_vac" => Dict("E" => 45.0, "F" => 3.0 , "V" => 1.0 ),
    "PH_hcp" => Dict("E" => 15.0, "F" => 50.0 , "V" => 1.0 ),
    "FLD_bcc" => Dict("E" => 100.0, "F" => 2.0 , "V" => 0.0 ),
    "FLD_hcp" => Dict("E" => 100.0, "F" => 2.0 , "V" => 0.0 ),
    )

Vref = OneBody(:Ti => -5.817622899211898)

IP, lsqinfo = lsqfit(dB, solver=(:rid, 1.05), weights = weights, Vref = Vref, asmerrs=true);

@show norm(lsqinfo["c"])
rmse_table(lsqinfo["errors"])   

save_dict(@__DIR__() * "/Ti_NRLTB_FLD_HMD_surf_vac_ev_ph_3_N4_18_DB30.json", Dict("IP" => write_dict(IP), "info" => lsqinfo))

al_img = IPFitting.Data.read_xyz("./exampleTB/bcc_2500_sel.xyz", energy_key="energy", force_key="force");

add_fits_serial!(IP, al_img, fitkey="IP")
rmse_, rmserel_ = rmse(al_img; fitkey="IP");
rmse_table(rmse_, rmserel_)


#save_dict(@__DIR__() * "/Ti_NRLTB_HMD_surf_vac_ev_ph_3_N3_20_DB30.json", Dict("IP" => write_dict(IP), "info" => lsqinfo))

# for i in 1:11
#     al = IPFitting.Data.read_xyz(@__DIR__() * "/DB_$(i-1).xyz", energy_key="energy", force_key="force");

#     dB = LsqDB("", B, al);

#     IP, lsqinfo = lsqfit(dB, solver=(:rrqr, 5e-7), weights = weights, Vref = Vref, asmerrs=true);

#     @show norm(lsqinfo["c"])
#     rmse_table(lsqinfo["errors"])   

#     save_dict(@__DIR__() * "/Ti_NRLTB_HMD_surf_vac_ev_ph_3_N3_16_DB$(i).json", Dict("IP" => write_dict(IP), "info" => lsqinfo))
# end






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