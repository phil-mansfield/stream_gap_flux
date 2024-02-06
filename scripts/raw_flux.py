import numpy as np
import matplotlib.pyplot as plt
import palette
import symlib
from colossus.cosmology import cosmology
import scipy.interpolate as interp
import scipy.optimize as optimize
from palette import pc
import scipy.special as special
from colossus.halo import mass_so

base_dir = "/sdf/home/p/phil1/ZoomIns"
suite = "SymphonyMilkyWay"
param = symlib.simulation_parameters(suite)
cosmo = cosmology.setCosmology('', symlib.colossus_parameters(param))
mp = param["mp"]/param["h100"]

use_dynamical_time = True

m_low = 1.5e8
m_high = 3e8
deg_conv = 41252.96125

def m23_S_moments(n_peak):
    z90 = 1.2816
    log_n = np.log10(n_peak)
    p9 = -0.3473 -0.3756*log_n
    p5 = -0.5054 -0.5034*log_n
    p1 = 0.0526 - 0.8121*log_n

    return p1, p5, p9

def m23_S(n_peak, mu):
    p1, p5, p9 = m23_S_moments(n_peak)

    log_mu = np.log10(mu)
    S = np.zeros(len(mu))
    low = log_mu < p5
    
    d_high = (log_mu - p5)/(p9 - p5)
    d_low = (log_mu - p5)/(p5 - p1)

    S[low] = (1+special.erf(d_low[low]*1.2816/np.sqrt(2)))/2
    S[~low] = (1+special.erf(d_high[~low]*1.2816/np.sqrt(2)))/2

    return S
   
def m23_weight(n_peak, mu):
    return 1/m23_S(n_peak, mu)

def linear_root(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    dx = -y1/m
    return x1+dx

def intersect_times(t, x, r):
    if len(t) < 2: return [[] for _ in range(len(r))]

    out = [None]*len(r)
    for i_r in range(len(r)):
        dr = np.sqrt(np.sum(x**2, axis=1)) - r[i_r]
        idx = np.where(dr[1:]*dr[:-1] < 0)[0]
        out[i_r] = linear_root(t[idx], dr[idx], t[idx+1], dr[idx+1])

    return out
    
def main():
    palette.configure(False)

    r_eval = np.array([25, 50, 100, 200])
    t_edges = np.linspace(1, 14, 13) - (14 - cosmo.age(0))

    cross_hist = np.zeros((len(r_eval), len(t_edges)-1))
    cross_hist_corr = np.zeros((len(r_eval), len(t_edges)-1))

    n_hosts = symlib.n_hosts(suite)
    rvir_ages = np.ones((len(r_eval), n_hosts))*np.nan

    for i_host in range(n_hosts):
        print(i_host)
        sim_dir = symlib.get_host_directory(base_dir, suite, i_host)

        z = 1/symlib.scale_factors(sim_dir) - 1
        h, _ = symlib.read_rockstar(sim_dir)
        s, hist = symlib.read_symfind(sim_dir)
        age = cosmo.age(z)
        t_dyn = mass_so.dynamicalTime(z, "vir")
        f_t_dyn = interp.interp1d(age, t_dyn)

        rvir_ages[:,i_host] = age[np.searchsorted(h["rvir"][0], r_eval)]

        for i_sub in range(1, len(s)):
            if hist["merger_ratio"][i_sub] > 0.1: continue

            ok = s["ok"][i_sub]
            if np.sum(ok) < 2: continue

            m = s["m"][i_sub,ok]
            log_m_func = interp.interp1d(age[ok], np.log10(m))
            def m_func(tt): return 10**log_m_func(tt)
            t = intersect_times(age[ok], s["x"][i_sub,ok], r_eval)

            n_peak = hist["mpeak"][i_sub]/mp

            for i_r in range(len(r_eval)):
                m = m_func(t[i_r])
                mu = m/hist["mpeak"][i_sub]
                ok = (m > m_low) & (m < m_high)
                if np.sum(ok) == 0: continue

                if use_dynamical_time:
                    w = f_t_dyn(t[i_r])[ok]
                else:
                    w = np.ones(len(t[i_r]))[ok]

                cross_hist[i_r] += np.histogram(t[i_r][ok], bins=t_edges,
                                                weights=w)[0]
                w = m23_weight(n_peak*np.ones(len(mu[ok])), mu[ok])
                if use_dynamical_time:
                    w *= f_t_dyn(t[i_r])[ok]
                cross_hist_corr[i_r] += np.histogram(
                    t[i_r][ok], bins=t_edges, weights=w)[0]

    cutoff_times = np.nanmedian(rvir_ages, axis=1)
    cross_hist = cross_hist/deg_conv
    cross_hist_corr = cross_hist_corr/deg_conv

    plt.figure()
    colors = [pc("r"), pc("o"), pc("b"), pc("p")]
    for i_r in range(len(r_eval)):
        t_mid = (t_edges[1:] + t_edges[:-1])/2
        ok = t_mid > cutoff_times[i_r]
        plt.plot(t_mid[ok], cross_hist[i_r,ok]/n_hosts,
                 c=colors[i_r],
                 label=r"$r=%d\,{\rm kpc}$" % r_eval[i_r])
        plt.plot(t_mid[ok], cross_hist_corr[i_r,ok]/n_hosts,
                 "--", lw=1.5,
                 c=colors[i_r])

    plt.legend(loc="upper right", fontsize=16)

    if use_dynamical_time:
        plt.ylabel(r"${\rm Subhalo\ flux}\ (N(r,\,t)/{\rm d}t/d\Omega\ ({\rm T_{\rm cross}}^{-1}) {deg}^{-2})$")
    else:
        plt.ylabel(r"${\rm Subhalo\ flux}\ (N(r,\,t)/{\rm d}t/d\Omega\ ({\rm Gyr}^{-1}) {deg}^{-2})$")
    plt.xlabel(r"$t\ ({\rm Gyr})$")
    plt.yscale("log")
    if use_dynamical_time:
        plt.ylim(1e-4, 1e-2)
    elif m_high > 1e8:
        plt.ylim(6e-5, 1e-3)
    elif suite == "SymphonyMilkyWayHR":
        plt.ylim(6e-4, 1e-2)
    else:
        plt.ylim(8e-5, 2e-3)
    plt.savefig("../plots/raw_flux.png")


if __name__ == "__main__":
    main()
