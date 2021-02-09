using Plots
using MDLearn
using IPFitting
using JuLIP
using ACE
using LinearAlgebra
using ASE
using JuLIP.MLIPs: SumIP

function plot_HMD(E_tot, E_pot, E_kin, T, P, i; k=50) # varEs,
    p1 = plot()
    plot!(p1,E_tot[2:end-k], label="")
    plot!(p1,E_kin[2:end-k], label="")
    plot!(p1,E_pot[2:end-k], label="")
    ylabel!(p1, "Energy (eV)")
    p2 = plot()
    plot!(p2, T[2:end-k],label="")
    ylabel!(p2, "T (K)")
    # p3 = plot()
    # plot!(p3, varEs[2:end-k],label="")
    # ylabel!(p3, "varE")
    p4 = plot()
    plot!(p4, P[2:end-k],label="")
    xlabel!(p4,"MDstep")
    ylabel!(p4, "P")
    p5 = plot(p1, p2, p4, size=(400,550), layout=grid(3, 1, heights=[0.6, 0.2, 0.2]))
    savefig("./exampleTB/HMD_standard_$(i).pdf")
end

function run_HMD(IP, B, S, c, c_samples, at; nsteps=100, temp=100, dt=0.1, τ=10.0)
    E_tot = zeros(nsteps)
    E_pot = zeros(nsteps)
    E_kin = zeros(nsteps)
    T = zeros(nsteps)
    P = zeros(nsteps)
    varEs = zeros(nsteps)
    varFs = zeros(nsteps)

    E0 = energy(IP, at)

    at = MDLearn.MD.MaxwellBoltzmann_scale(at, temp)
    at = MDLearn.MD.Stationary(at)

    cfgs = []

    Vref = OneBody(:Ti => -5.817622899211898)

    running = true

    i = 2
    while running && i < nsteps
        #at, p = MDLearn.HMD.VelocityVerlet_uncertain(IP, B, at, dt * MDLearn.MD.fs, S, c, τ=τ)
        at = MDLearn.MD.VelocityVerlet(IP, at, dt * MDLearn.MD.fs)
        #meanE, varE, meanF, varF = MDLearn.HMD.get_energy_forces(Vref, B, c_samples, at);
        #varEs[i] = varE
        Ek = ((0.5 * sum(at.M) * norm(at.P ./ at.M)^2)/length(at.M)) / length(at.M)
        Ep = (energy(IP, at) - E0) / length(at.M)
        E_tot[i] = Ek + Ep
        E_pot[i] = Ep
        E_kin[i] = Ek
        T[i] = Ek / (1.5 * MDLearn.MD.kB)
        i+=1
        if i % 10 == 0
            @show abs((E_tot[i-1]/E_tot[2] - 1.0))
            if abs((E_tot[i-1]/E_tot[2] - 1.0)) > 0.1
                running = false
            end
            τ *= 1.05
            push!(cfgs, ASEAtoms(at))
        end
    end
    
    return E_tot[1:i], E_pot[1:i], E_kin[1:i], T[1:i], P[1:i], varEs[1:i], varFs[1:i], cfgs
end

function do_fit(al)
    r0 = rnn(:Ti)
    
    Bpair = pair_basis(species = :Ti,
        r0 = r0,
        maxdeg = 3,
        rcut = 7.0,
        pcut = 1,
        pin = 0)
    
    Bsite = rpi_basis(species = :Ti,
          N = 3,                       # correlation order = body-order - 1
            maxdeg = 10,            # polynomial degree
          r0 = r0,                      # estimate for NN distance
          #D = SparsePSHDegree(; wL=1.3, csp=1.0),
          rin = 0.70*r0, rcut = 5.5,   # domain for radial basis (cf documentation) #5.5
          pin = 2);

    weights = Dict(
        "ignore"=> [],
        "default" => Dict("E" => 10.0, "F" => 1.0 , "V" => 1.0 ),
        )

    Vref = OneBody(:Ti => -5.817622899211898)

    B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);
    
    dB = LsqDB("", B, al);
    
    Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
                                Vref=Vref, Ibasis = :,Itrain = :,
                                weights=weights, regularisers = [])
    
    m, S = MDLearn.Uncertain.posterior(Ψ, Y, 5.0, 5.0);
    c_samples = MDLearn.Uncertain.do_brr(Ψ, Y, 5.0, 5.0, 20);
    #IP, lsqinfo = lsqfit(dB, weights = weights, Vref = Vref, asmerrs=true);
    IP = JuLIP.MLIPs.SumIP(Vref, JuLIP.MLIPs.combine(B, c_samples[:,1]))

    add_fits_serial!(IP, al, fitkey="IP")
    rmse_, rmserel_ = rmse(al; fitkey="IP");
    rmse_table(rmse_, rmserel_)
    
    return IP, B, S, c_samples[:,1], c_samples
end

E_errs = []
F_errs = []

for i in 1:15
    al = IPFitting.Data.read_xyz(@__DIR__() * "/DB_standard_$(i-1).xyz", energy_key="energy", force_key="force");
    IP, B, S, c, c_samples = do_fit(al)
    
    al_liq = IPFitting.Data.read_xyz(@__DIR__() * "/bcc_2500_sel.xyz", energy_key="energy", force_key="force");
    add_fits_serial!(IP, al_liq, fitkey="IP")
    rmse_, rmserel_ = rmse(al_liq; fitkey="IP");
    rmse_table(rmse_, rmserel_)

    push!(E_errs, rmse_["set"]["E"])
    push!(F_errs, rmse_["set"]["F"])

    E_tot, E_pot, E_kin, T, P, varEs, varFs, cfgs = run_HMD(IP, B, S, c, c_samples, bulk(:Ti, cubic=true) * (2,2,2), nsteps=1000, temp=8050);
    plot_HMD(E_tot, E_pot, E_kin, T, P, i, k=1)
    write_xyz("./exampleTB/crash_$(i).xyz", cfgs[end])
    run(`/Users/Cas/anaconda2/bin/python /Users/Cas/.julia/dev/MDLearn/exampleTB/convert.py $(i)`)
end