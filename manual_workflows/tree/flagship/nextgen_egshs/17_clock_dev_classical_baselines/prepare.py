import numpy as np

def rz(a):return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])

def orbit(t,j,k):
    a=6378137.+550000.;n=np.sqrt(3.986004418e14/a**3);w=7.292115e-5
    inc=np.deg2rad(53);rx=np.array([[1,0,0],[0,np.cos(inc),-np.sin(inc)],[0,np.sin(inc),np.cos(inc)]])
    f=2*np.pi*k/20+2*np.pi*j/240+n*t
    p=rz(2*np.pi*j/12)@rx@np.array([a*np.cos(f),a*np.sin(f),0])
    v=rz(2*np.pi*j/12)@rx@np.array([-a*n*np.sin(f),a*n*np.cos(f),0])
    pe=rz(-w*t)@p;ve=rz(-w*t)@v-np.cross([0,0,w],pe)
    return pe,ve
