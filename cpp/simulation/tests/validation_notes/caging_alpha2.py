import numpy as np
R, TAU, L, CUT = 49.0, 10000.0, 916.0, 132.8
def load(p):
    t,c,x,y=[],[],[],[]
    for line in open(p):
        if line[0]=="#": continue
        f=line.split(); t.append(float(f[0])); c.append(int(f[1]))
        x.append(float(f[2])); y.append(float(f[3]))
    t=np.array(t);c=np.array(c);x=np.array(x);y=np.array(y)
    ids=np.unique(c); tm=np.unique(t); nt,nc=len(tm),len(ids)
    X=np.empty((nt,nc));Y=np.empty((nt,nc))
    ti={v:i for i,v in enumerate(tm)}; ci={v:i for i,v in enumerate(ids)}
    for k in range(len(t)):
        X[ti[t[k]],ci[c[k]]]=x[k]; Y[ti[t[k]],ci[c[k]]]=y[k]
    return tm,X,Y
def unwrap(A):
    o=np.empty_like(A);o[0]=A[0]
    for i in range(1,len(A)):
        d=A[i]-A[i-1]; d-=L*np.round(d/L); o[i]=o[i-1]+d
    return o
def nbrs(X,Y,f):
    dx=X[f][:,None]-X[f][None,:]; dy=Y[f][:,None]-Y[f][None,:]
    dx-=L*np.round(dx/L); dy-=L*np.round(dy/L)
    d2=dx*dx+dy*dy; np.fill_diagonal(d2,np.inf); return d2<CUT*CUT
for tag in ("ctrl","soft"):
    tm,X,Y=load(f"/scratch/project_2017848/ssilber/review_bench/long/{tag}.txt")
    dt=tm[1]-tm[0]; XU,YU=unwrap(X),unwrap(Y); nt=len(tm)
    lags=np.unique(np.geomspace(1,nt//2,30).astype(int))
    print(f"=== {tag}  dt={dt:.0f} TU  frames={nt}  max lag={lags[-1]*dt/TAU:.2f} tau ===")
    print(f"   lag/tau     alpha2      CR-MSD      absMSD    CR/abs")
    rows=[]
    for lag in lags:
        norig=min(40, max(4, (nt-lag)//50))
        origins=np.linspace(0,nt-lag-1,norig).astype(int)
        s2=[];ab=[]
        for f0 in origins:
            nb=nbrs(X,Y,f0); z=nb.sum(1); m=z>0
            dx=XU[f0+lag]-XU[f0]; dy=YU[f0+lag]-YU[f0]
            mx=(nb*dx[None,:]).sum(1)/np.maximum(z,1); my=(nb*dy[None,:]).sum(1)/np.maximum(z,1)
            cx=(dx-mx)[m]; cy=(dy-my)[m]
            s2.append(cx*cx+cy*cy); ab.append(dx[m]**2+dy[m]**2)
        s2=np.concatenate(s2); ab=np.concatenate(ab)
        a2=np.mean(s2*s2)/(2*np.mean(s2)**2)-1
        rows.append((lag*dt/TAU,a2,np.mean(s2),np.mean(ab),s2))
        print(f"{lag*dt/TAU:10.4f} {a2:10.4f} {np.mean(s2):11.3f} {np.mean(ab):11.3f} {np.mean(s2)/np.mean(ab):8.4f}")
    # peak over the physically meaningful range only (CR-MSD > 0.01 px^2)
    valid=[r for r in rows if r[2]>0.01]
    pk=max(valid,key=lambda r:r[1])
    print(f"  -> alpha_2 max over valid range: {pk[1]:.4f} at {pk[0]:.4f} tau")
    d=np.sqrt(pk[4]); hi=np.percentile(d,99.7)
    h,e=np.histogram(d,bins=120,range=(0,hi)); rc=0.5*(e[1:]+e[:-1])
    G=h/np.maximum(rc,1e-9); Gs=np.convolve(G,np.ones(7)/7,mode="same")
    p0=int(np.argmax(Gs)); mn=None
    for i in range(p0+3,len(Gs)-3):
        if Gs[i]<=Gs[i-1] and Gs[i]<=Gs[i+1]: mn=rc[i]; break

    lab = f"{mn:.2f} px = {mn/(2*R):.3f} diam" if mn else "none (unimodal)"
    print(f"  -> van Hove(cage-rel) at that lag: peak {rc[p0]:.2f} px, first min {lab}")