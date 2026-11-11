"""Exact radial AC backward/forward sweep, checked independently against pandapower.

Fixed discrete commands permit reuse of solved states across controllers. No
linearized or interpolated state is used for the reported violation labels.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def solve_radial(parent,z,p,q,rating,battery_bus,command,gen_buses,curtail):
    n=len(p);v=np.ones(n,np.complex128);s=p+1j*q
    s=s.copy();s[battery_bus]-=command
    for b in gen_buses:s[b]+=curtail/len(gen_buses)
    branches=np.zeros(n,np.complex128);converged=False
    # Units: MW/Mvar on a 1-MVA base, impedance on the corresponding base.
    for it in range(100):
        previous=v.copy();current=np.conj(s/v)
        for b in range(n-1,0,-1):
            branches[b]=current[b];current[parent[b]]+=current[b]
        for b in range(1,n):v[b]=v[parent[b]]-z[b]*branches[b]
        if np.max(np.abs(v-previous))<1e-10:converged=True;break
        if np.min(np.abs(v))<.1:break
    vm=np.abs(v);vmin=vm.min();vmax=vm.max()
    loading=0.
    for b in range(1,n):loading=max(loading,abs(branches[b])/rating[b])
    grid=(v[0]*np.conj(np.sum(s/v))).real
    feasible=converged and vmin>=.95-1e-7 and vmax<=1.05+1e-7 and loading<=1+1e-7
    severity=max(0.,.95-vmin)+max(0.,vmax-1.05)+max(0.,loading-1.)
    if not converged:severity=max(severity,1.)
    return np.array([1.*feasible,vmin,vmax,loading,severity,1.*converged,grid])

def network(name='two_bus',rating_factor=1.,load_multiplier=1.,pv_capacity=1.8):
    if name=='two_bus':
        voltage=20.;parent=np.array([0,0],dtype=np.int64)
        z=np.array([0,(.42+.16j)*3/(voltage**2)],complex)
        pl=np.array([0.,.6]);ql=pl*.28;rg=np.array([0.,.4])
        rating=np.array([1.,.035*np.sqrt(3)*voltage])*rating_factor
        return {'name':name,'parent':parent,'z':z,'load':pl,'reactive':ql,'gen':rg,'rating':rating,'battery_bus':1,'gen_buses':np.array([1],np.int64),'pmax':.26,'capacity':.8,'voltage':voltage}
    import pandapower.networks as pn
    net=pn.case33bw();voltage=12.66
    parent=np.zeros(33,np.int64);z=np.zeros(33,complex)
    for r in net.line[net.line.in_service].itertuples():
        parent[r.to_bus]=r.from_bus;z[r.to_bus]=(r.r_ohm_per_km+1j*r.x_ohm_per_km)*r.length_km/voltage**2
    # case33bw ordering is topological, required by the sweep.
    assert np.all(parent[1:]<np.arange(1,33))
    pl=np.zeros(33);ql=pl.copy()
    for r in net.load.itertuples():pl[r.bus]+=r.p_mw;ql[r.bus]+=r.q_mvar
    pl*=load_multiplier;ql*=load_multiplier
    rg=np.zeros(33);rg[[17,24,32]]=pv_capacity/3
    # Thermal ratings are imposed study settings; the source has no usable ratings.
    rating=np.full(33,.4*np.sqrt(3)*voltage*rating_factor)
    return {'name':name,'parent':parent,'z':z,'load':pl,'reactive':ql,'gen':rg,'rating':rating,'battery_bus':17,'gen_buses':np.array([17,24,32],np.int64),'pmax':.8,'capacity':3.2,'voltage':voltage}

@njit(cache=True)
def tables_kernel(load,ren,parent,z,pl,ql,rg,rating,bus,gb,pmax,task):
    n=len(load);out=np.empty((n,17,7))
    for i in range(n):
        p=pl*(load[i]/600.)-rg*(ren[i]/400.);q=ql*(load[i]/600.)
        for k in range(17):
            u=(k-8)*pmax/8
            curtail=0.
            if task==1:
                u=0.;curtail=k*rg.sum()/16
                curtail=min(curtail,rg.sum()*max(ren[i],0.)/400.)
            out[i,k]=solve_radial(parent,z,p,q,rating,bus,u,gb,curtail)
    return out

def physics_tables(values,net,task='battery'):
    return tables_kernel(values[:,0],values[:,1],net['parent'],net['z'],net['load'],net['reactive'],net['gen'],net['rating'],net['battery_bus'],net['gen_buses'],net['pmax'],int(task=='pv'))

def validate_against_pandapower(seed=22355,n=120):
    import pandapower as pp
    import pandapower.networks as pn
    rng=np.random.default_rng(seed);rows=[]
    for name in ['two_bus','ieee33']:
        spec=network(name)
        if name=='two_bus':
            net=pp.create_empty_network();pp.create_bus(net,vn_kv=20);pp.create_bus(net,vn_kv=20);pp.create_ext_grid(net,0)
            pp.create_line_from_parameters(net,0,1,length_km=3,r_ohm_per_km=.42,x_ohm_per_km=.16,c_nf_per_km=0,max_i_ka=.035)
            pp.create_load(net,1,p_mw=.6,q_mvar=.168);pp.create_sgen(net,1,p_mw=.4)
        else:
            net=pn.case33bw();net.line.max_i_ka=.4
            for b in [17,24,32]:pp.create_sgen(net,b,p_mw=.6)
        battery=pp.create_sgen(net,spec['battery_bus'],p_mw=0)
        basep=net.load.p_mw.to_numpy().copy();baseq=net.load.q_mvar.to_numpy().copy()
        genidx=net.sgen.index[net.sgen.index!=battery];baseg=net.sgen.loc[genidx,'p_mw'].to_numpy().copy()
        for i in range(n):
            l=rng.uniform(.45,1.3);g=rng.uniform(0,1.8);u=rng.integers(-8,9)*spec['pmax']/8
            net.load.loc[:,'p_mw']=basep*l;net.load.loc[:,'q_mvar']=baseq*l
            net.sgen.loc[genidx,'p_mw']=baseg*g;net.sgen.at[battery,'p_mw']=u
            pp.runpp(net,numba=True,init='flat',tolerance_mva=1e-10,max_iteration=100)
            a=solve_radial(spec['parent'],spec['z'],spec['load']*l-spec['gen']*g,spec['reactive']*l,spec['rating'],spec['battery_bus'],u,spec['gen_buses'],0.)
            actual=np.array([net.res_bus.vm_pu.min(),net.res_bus.vm_pu.max(),net.res_line.loading_percent.max()/100])
            err=np.max(abs(a[1:4]-actual));rows.append({'network':name,'case':i,'max_error':err})
            if err>1e-6:raise AssertionError((name,i,a,actual))
    return rows
