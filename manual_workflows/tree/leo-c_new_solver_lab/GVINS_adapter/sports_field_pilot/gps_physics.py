"""GPS-only exact equation port and independently differentiated QA expressions.

Source: pinned gnss_comm/src/gnss_utility.cpp, GPS branches, GPLv3/RTKLIB
notices retained in the bound repository. This file does not change that source.
Time arguments are seconds relative to toe to avoid large-epoch cancellation.
No receiver position estimation, candidate fitting, or selector is implemented.
"""
import math
import numpy as np
C=299792458.0
MU=3.986005e14
OMEGA=7.2921151467e-5
STEPS=(0.01,0.1,1.0)

def kepler(m,e):
    value=m;previous=1e6;n=0
    while n<30 and abs(value-previous)>1e-14:
        previous=value
        value-=(value-e*math.sin(value)-m)/(1-e*math.cos(value));n+=1
    if n==30: raise ArithmeticError('KEPLER_ITERATION_LIMIT')
    return previous # Exact official return identity (not final Newton value).

def orbit(eph, seconds_from_toe):
    tk=seconds_from_toe
    if tk>302400: tk-=604800
    elif tk< -302400: tk+=604800
    e=eph['e'];a=eph['A'];n=math.sqrt(MU/a**3)+eph['delta_n']
    ek=kepler(eph['M0']+n*tk,e);se=math.sin(ek);ce=math.cos(ek)
    ed=n/(1-e*ce);vd=math.sqrt(1-e*e)*ed/(1-e*ce)
    phi=math.atan2(math.sqrt(1-e*e)*se,ce-e)+eph['omg']
    s2=math.sin(2*phi);c2=math.cos(2*phi)
    u=phi+eph['cus']*s2+eph['cuc']*c2
    r=a*(1-e*ce)+eph['crs']*s2+eph['crc']*c2
    inc=eph['i0']+eph['i_dot']*tk+eph['cis']*s2+eph['cic']*c2
    ud=vd+2*vd*(eph['cus']*c2-eph['cuc']*s2)
    rd=a*e*ed*se+2*vd*(eph['crs']*c2-eph['crc']*s2)
    ind=eph['i_dot']+2*vd*(eph['cis']*c2-eph['cic']*s2)
    su=math.sin(u);cu=math.cos(u);si=math.sin(inc);ci=math.cos(inc)
    x=r*cu;y=r*su;xd=rd*cu-r*ud*su;yd=rd*su+r*ud*cu
    od=eph['OMG_dot']-OMEGA
    omg=eph['OMG0']+od*tk-OMEGA*eph['toe_tow'];so=math.sin(omg);co=math.cos(omg)
    p=np.array([x*co-y*ci*so,x*so+y*ci*co,y*si])
    t1=xd-y*od*ci;t2=x*od+yd*ci-y*ind*si
    official=np.array([t1*co-t2*so,t1*so+t2*co,yd*si+yd*ind*ci])
    # Independent chain-rule derivative of z = y_orbit * sin(inclination).
    derivative=np.array([t1*co-t2*so,t1*so+t2*co,yd*si+y*ind*ci])
    dt=seconds_from_toe+(eph['toe_ns']-eph['toc_ns'])/1e9
    relativistic=-2*math.sqrt(MU*a)*e*se/C**2
    clock=eph['af0']+eph['af1']*dt+eph['af2']*dt*dt+relativistic
    drift=eph['af1']+2*eph['af2']*dt-2*math.sqrt(MU*a)*e*ce*ed/C**2
    return p,official,derivative,clock,drift

def svdt(eph, seconds_from_toe):
    dt=seconds_from_toe+(eph['toe_ns']-eph['toc_ns'])/1e9
    for _ in range(2): dt-=eph['af0']+eph['af1']*dt+eph['af2']*dt*dt
    return eph['af0']+eph['af1']*dt+eph['af2']*dt*dt

def finite_difference(eph,t,h):
    plus=orbit(eph,t+h);minus=orbit(eph,t-h)
    return (plus[0]-minus[0])/(2*h),(plus[3]-minus[3])/(2*h)

def rotation(flight):
    a=OMEGA*flight;ca=math.cos(a);sa=math.sin(a)
    return np.array([[ca,sa,0],[-sa,ca,0],[0,0,1]])

def jz(p): return np.array([-p[1],p[0],0.0])

def earth_rotation_forms(s,v,r,vr):
    """Geometric-flight rotation plus explicit first-order range corrections.

    'epoch' differentiates both states with respect to a common epoch while
    holding the emission offset fixed (the pinned GVINS Doppler convention).
    'retarded' additionally uses dt_tx/dt_rx=1-range_rate/c. Both pairs retain
    the derivative of the geometric flight angle, unlike rotate-only adapter.
    """
    d=s-r;rho=np.linalg.norm(d);u=d/rho;g0=float(u@(v-vr))
    tau=rho/C
    for _ in range(5):
        rot=rotation(tau);sr=rot@s;tau=np.linalg.norm(sr-r)/C
    rot=rotation(tau);sr=rot@s;vsr=rot@v;ur=(sr-r)/np.linalg.norm(sr-r)
    numerator=float(ur@(vsr-vr));angle_term=float(ur@(OMEGA*jz(sr)))/C
    form_a_epoch=numerator/(1+angle_term)
    form_a_retarded=numerator/(1+angle_term+float(ur@vsr)/C)
    sat_term=OMEGA/C*(v[0]*r[1]-v[1]*r[0])
    receiver_term=OMEGA/C*(s[0]*vr[1]-s[1]*vr[0])
    form_b_epoch=g0+sat_term+receiver_term
    form_b_retarded=form_b_epoch/(1+(float(u@v)+sat_term)/C)
    return {'form_a_epoch_mps':form_a_epoch,'form_b_epoch_mps':form_b_epoch,
            'form_a_retarded_mps':form_a_retarded,'form_b_retarded_mps':form_b_retarded,
            'naive_rotated_mps':numerator,'unrotated_mps':g0,
            'geometric_flight_s':tau,'sagnac_satellite_term_mps':sat_term,
            'sagnac_receiver_term_mps':receiver_term,'rotated_los':ur,'rotated_satellite_position':sr}

def corrected_distance(eph,tx0,r,vr,dt):
    """Independent implicit light-cone distance for derivative QA, no fitting."""
    s0=orbit(eph,tx0)[0];rho0=np.linalg.norm(s0-r)
    for _ in range(8): rho0=np.linalg.norm(rotation(rho0/C)@s0-r)
    rho=rho0
    for _ in range(8):
        st=orbit(eph,tx0+dt-(rho-rho0)/C)[0]
        rho=np.linalg.norm(rotation(rho/C)@st-(r+dt*vr))
    return rho
