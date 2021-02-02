using Plots
using MDLearn

function plot_HMD(E_tot, E_pot, E_kin, T, varEs, i; k=50)
    p1 = plot()
    plot!(p1,E_tot[2:end-k], label="Etot", legend=:top)
    plot!(p1,E_kin[2:end-k], label="Ekin")
    plot!(p1,E_pot[2:end-k], label="Epot")
    ylabel!(p1, "Energy (eV)")
    p2 = plot()
    plot!(p2, T[2:end-k],label="")
    ylabel!(p2, "T (K)")
    p3 = plot()
    plot!(p3, varEs[2:end-k],label="")
    xlabel!(p3,"MDstep")
    ylabel!(p3, "varE")
    p5 = plot(p1, p2, p3, layout=grid(3, 1, heights=[0.6, 0.2, 0.2]))
    savefig("HMD_$(i).pdf")
end

function run_HMD(IP, B, S, c, at; nsteps=100, temp=100, dt=1.0, τ=1e-10)
    E_tot = zeros(nsteps)
    E_pot = zeros(nsteps)
    E_kin = zeros(nsteps)
    T = zeros(nsteps)
    varEs = zeros(nsteps)
    varFs = zeros(nsteps)

    E0 = energy(IP, at)

    at = MDLearn.MD.MaxwellBoltzmann_scale(at, temp)
    at = MDLearn.MD.Stationary(at)

    cfgs = []

    i = 2
    while T[i-1] < 5050 && i < nsteps
        at = VelocityVerlet(IP, at, dt * fs)
        if i % 10 == 0
            at = VelocityVerlet_uncertain2(IP, B, at, dt * fs, S, c, τ=τ)
            meanE, varE, meanF, varF = get_energy_forces(B, c_samples, at);
            varEs[i] = varE
            varFs[i] = sqrt(mean(vcat(varF...).^2))
        end
        if i % 100 == 0
            if p < 3.0
                p *= 1.05
            end
        end
        Ek = ((0.5 * sum(at.M) * norm(at.P ./ at.M)^2)/length(at.M)) / length(at.M)
        Ep = (energy(IP, at) - E0) / length(at.M)
        E_tot[i] = Ek + Ep
        E_pot[i] = Ep
        E_kin[i] = Ek
        T[i] = Ek / (1.5 * kB)
        i+=1
        if i % 10 == 0
            push!(cfgs, ASEAtoms(at))
        end
    end
    
    return E_tot[1:i], E_pot[1:i], E_kin[1:i], T[1:i], varEs[1:i], varFs[1:i], cfgs
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
          N = 2,                       # correlation order = body-order - 1
            maxdeg = 10,            # polynomial degree
          r0 = r0,                      # estimate for NN distance
          #D = SparsePSHDegree(; wL=1.3, csp=1.0),
          rin = 0.50*r0, rcut = 5.5,   # domain for radial basis (cf documentation) #5.5
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
    
    m, S = posterior(Ψ, Y, 200.0, 2.0);
    IP, lsqinfo = lsqfit(dB, solver=(:rid, 1.05), weights = weights, Vref = Vref, asmerrs=true);
    
    rmse_table(lsqinfo["errors"])
    
    return IP, B, S, lsqinfo["c"]
end


for i in 1:10
    al = IPFitting.Data.read_xyz("./DB_$(i-1).xyz", energy_key="energy", force_key="force");
    IP, B, S, c = do_fit(al)
    
    E_tot, E_pot, E_kin, T, varEs, varFs, cfgs = run_HMD(IP, B, S, c, bulk(:Ti, cubic=true) * (2,2,2), nsteps=1000, temp=550);
    plot_HMD(E_tot, E_pot, E_kin, T, varEs, varFs, i, k=1)
    write_xyz("crash_$(i).xyz", cfgs[end])
    run(`/Users/Cas/anaconda2/bin/python convert.py $(i)`)
end