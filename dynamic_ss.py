import numpy as np
from scipy.stats import norm
def dynamic_safety_stock(demand_history, lead_time, service_level=0.95, window=28, min_window=7):
    z=norm.ppf(service_level); n=len(demand_history)
    results=[]
    for t in range(min_window,n):
        w=min(window,t)
        recent=demand_history[t-w:t]
        mu=np.mean(recent); sig=np.std(recent,ddof=1)
        trend=np.polyfit(range(w),recent,1)[0]
        vol_ratio=sig/max(mu,1)
        adaptive_z=z*(1+0.5*min(vol_ratio,1))
        ss=adaptive_z*sig*np.sqrt(lead_time)
        rop=mu*lead_time+ss
        results.append({"period":t,"demand_avg":round(mu,1),"demand_std":round(sig,1),
                        "trend":round(trend,2),"safety_stock":round(ss,0),"reorder_point":round(rop,0)})
    return results
if __name__=="__main__":
    rng=np.random.default_rng(42)
    stable=rng.normal(100,10,30).tolist()
    volatile=rng.normal(120,40,30).tolist()
    demand=stable+volatile
    results=dynamic_safety_stock(demand,lead_time=7)
    print("Stable period SS:",results[20]["safety_stock"])
    print("Volatile period SS:",results[50]["safety_stock"])
