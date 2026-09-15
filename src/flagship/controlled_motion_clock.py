"""V27 declared truth domain only. No candidate or probe imports this module."""
import numpy as np, ast
from pathlib import Path
def orbit_functions(base):
    functions={}
    for rel,names in [('17_clock_dev_classical_baselines/prepare.py',['rz','orbit']),('20_matched_observation_support_experiment/prepare.py',['rz','orbit_add'])]:
        path=Path(base)/rel;tree=ast.parse(path.read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes}==set(names)
        scope={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
        functions.update({n:scope[n] for n in names if n!='rz'})
    return functions
def trajectory(t,anchor,east,motion):
    t=np.asarray(t,float);q=np.asarray(anchor)+[500.,-300.,200.];omega=2*np.pi/30
    if motion=='CV5':
        speed=np.full(t.shape,5.);position=5*(t-30);acc=np.zeros(t.shape)
    elif motion in ('SIN_P45','SIN_P135'):
        phase={'SIN_P45':np.pi/4,'SIN_P135':3*np.pi/4}[motion]
        speed=5*(1+.35*np.sin(omega*t+phase))
        position=5*((t-30)+.35*(np.cos(omega*30+phase)-np.cos(omega*t+phase))/omega)
        acc=1.75*omega*np.cos(omega*t+phase)
    elif motion=='TRIANGULAR_SPEED_STRESS':
        f=np.where(t<=6,0,np.where(t<=18,(t-6)/12,np.where(t<=27,(27-t)/9,0)))
        F=np.where(t<=6,0,np.where(t<=18,(t-6)**2/24,np.where(t<=27,6+(t-18)-(t-18)**2/18,10.5)))
        speed=5+3.5*(f-.35);position=5*(t-30)+3.5*(F-10.5-.35*(t-30))
        acc=np.where(t<=6,0,np.where(t<=18,3.5/12,np.where(t<=27,-3.5/9,0)))
    else:raise ValueError(motion)
    return q,q+position[:,None]*east,speed[:,None]*east,acc
def signal(g,motion,b0,bdot):
    q,r,v,acc=trajectory(g['tau'],g['anchor'],g['east'],motion)
    delta=g['sat']-r;u=delta/np.linalg.norm(delta,axis=1)[:,None]
    clock=b0+bdot*g['tau'];y=np.sum(u*(g['sv']-v),axis=1)+clock
    _,rp,vp,ap=trajectory(np.array([0.,10.,20.,30.]),g['anchor'],g['east'],motion)
    assert np.allclose(rp[-1],q,atol=1e-9,rtol=0)
    assert np.allclose(rp[-1]-rp[0],150*g['east'],atol=2e-9,rtol=0)
    stats=[dict(window='W'+str(30-lo),displacement_east_m=float((rp[-1]-rp[i])@g['east']),mean_velocity_east_mps=float((rp[-1]-rp[i])@g['east'])/(30-lo)) for i,lo in enumerate([0,10,20])]
    return y,dict(q=q,position=r,velocity=v,velocity_at_output=vp[-1],clock=clock,b0=b0,bdot=bdot,output_epoch=g['output_epoch'],window_motion=stats,endpoint_acceleration_mps2=ap[-1])
