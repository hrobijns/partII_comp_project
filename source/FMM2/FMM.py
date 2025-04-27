import numpy as np
import scipy.special as sspecial
from tree import LibFmm2dNode
from particles import Particles2D
from copy import deepcopy

#the actual sim box:
bbox = np.array([[0.0, 0.0], [35, 35]])
#where we want particles setup
pbbox = np.array([[10.0, 10.0], [25, 25]])
dt = 0.01

Ntot = 1000
#charges/masses of type 0 and type 1
charges = [1, -1]
masses = [1836, 1]
ids = np.arange(Ntot)
types = np.ones(Ntot, dtype='int')
types[int(Ntot/2):] = 0

#generate positions in the box
rs = np.random.rand(Ntot, 2)
for i in range(2):
    rs[:, i] = (pbbox[1, i] - pbbox[0, i])*rs[:, i] + pbbox[0, i]

#kinetic energies corresponding to ~13.5 ev.
vs = np.zeros((Ntot, 2))
for i in range(2):
    vs[:int(Ntot/2), i] = np.random.randn(int(Ntot/2))*np.sqrt(0.5/masses[0])
    vs[int(Ntot/2):, i] = np.random.randn(int(Ntot/2))*np.sqrt(0.5/masses[1])
    
    
all_particles = Particles2D.from_vecs(bbox, charges, masses, ids, types, rs, vs, dt)
fmm_part = deepcopy(all_particles)
exact_part = deepcopy(all_particles)

p = 15
n = 7
c = 1

def get_leaf_mpole2d(c, charges, parray, l):
    out = 0
    if l == 0:
        for i in range(parray.shape[0]):
            out += charges[parray['types'][i]]
    else:
        c = c[0] + 1j*c[1]
        for i in range(parray.shape[0]):
            omega_n = parray['x'][i] + 1j*parray['y'][i]
            omega_n = (omega_n - c)**l
            omega_n = omega_n*charges[parray['types'][i]]/l
            out += omega_n
    return out


def set_leaf_mpoles2d(td, charges):
    max_d = len(td.keys()) - 1
    for k in td[max_d].keys():
        for l in range(td[max_d][k].mpoles.shape[0]):
            td[max_d][k].mpoles[l] = get_leaf_mpole2d(td[max_d][k].center, charges, td[max_d][k].parray, l)

#holds all nodes at each level
tree_dict = {}
for l in range(n):
    tree_dict[l] = {}
    
#liberal in sense that every node stores its center, children, parents, and neighbors.
root = LibFmm2dNode(tree_dict, fmm_part.parray, n, p)
set_leaf_mpoles2d(tree_dict, fmm_part.charges)

##UPWARD PASS:
#get lists of coefficients and powers
def get_m2m_lists(mnum):
    coeffs_list = []
    pows_list = []
    for lval in range(mnum):
        if lval == 0:
            coeffs = np.array([1])
            pows = np.array([0])
            coeffs_list.append(coeffs)
            pows_list.append(pows)
        else:
            ls = np.full((lval+1,), lval, dtype=np.float64)
            ms = np.arange(lval+1) 
            coeffs = sspecial.binom(ls-1, ms-1)
            coeffs[0] = 1.0/lval
            coeffs_list.append(coeffs)
            pows = ls - ms
            pows_list.append(pows)
    return coeffs_list, pows_list


def get_m2m(node, z0, l, lcoeffs, lpows):
    out = np.sum(lcoeffs*node.mpoles[:l+1]*(z0**lpows))
    return out
        
    
def upward_pass(td):
    cl, pl = get_m2m_lists(td[0][0].mpoles.shape[0])
    lev_list = list(td.keys())
    lev_list.reverse()
    #won't pass up for level 0
    lev_list = lev_list[:-1]
    for l in lev_list:
        for k in td[l].keys():
            par = td[l][k].parent
            z0 = td[l][k].center[0]-td[l-1][par].center[0]
            z0 = z0 + 1j*(td[l][k].center[1] - td[l-1][par].center[1])
            for p in range(len(cl)):
                td[l-1][par].mpoles[p] += get_m2m(td[l][k], z0, p, cl[p], pl[p])

upward_pass(tree_dict)


## DOWNARD PASS:
#get lists of coefficients and powers
def get_m2l_lists(mnum):
    coeffs_list = []
    pows_list = []
    for lval in range(mnum):
        if lval == 0:
            coeffs = np.array([1]*(mnum-1))
            pows = -1*np.array(range(1, mnum))
            coeffs_list.append(coeffs)
            pows_list.append(pows)
        else:
            ls = np.full((mnum,), lval, dtype=np.float64)
            ms = np.arange(mnum)
            coeffs = sspecial.binom(ls+ms-1, ms-1)
            coeffs = coeffs*np.power(-1, ms)
            coeffs[0] = 1.0/lval
            coeffs_list.append(coeffs)
            pows = -1*ms
            pows_list.append(pows)
    return coeffs_list, pows_list


def get_m2l_term(node, z0, l, lcoeffs, lpows):
    out = 0
    if l==0:
        out += -1*node.mpoles[0]*np.log(-1*z0)
        out += np.sum(node.mpoles[1:]*np.power(z0, lpows))
    else:
        out += np.sum(node.mpoles*lcoeffs*np.power(z0, lpows))
        out = out/np.power(z0, l)
    return out

#l=local
#f=far
def get_m2l(lnode, fnode, clst, plst):
    z0 = fnode.center[0] - lnode.center[0]
    z0 = z0 + 1j*(fnode.center[1]-lnode.center[1])
    for p in range(len(clst)):
        lnode.lexp[p] += get_m2l_term(fnode, z0, p, clst[p], plst[p])

#get list of coefficients and powers
def get_l2l_lists(mnum):
    coeffs_list = []
    pows_list = []
    for lval in range(mnum):
        ms = np.array(range(lval, mnum))
        coeffs = sspecial.binom(ms, lval)
        pows = ms-lval
        coeffs_list.append(coeffs)
        pows_list.append(pows)
    return coeffs_list, pows_list


def get_l2l_term(node, z0, l, lcoeffs, lpows):
    return np.sum(node.lexp[l:]*lcoeffs*np.power(-1*z0, lpows))

#c=child
#p=parent
def get_l2l(cnode, pnode, clst, plst):
    z0 = pnode.center[0] - cnode.center[0]
    z0 = z0 + 1j*(pnode.center[1] - cnode.center[1])
    for p in range(len(clst)):
        cnode.lexp[p] += get_l2l_term(pnode, z0, p, clst[p], plst[p])

def downward_pass(td):
    cl1, pl1 = get_m2l_lists(td[0][0].mpoles.shape[0])
    cl2, pl2 = get_l2l_lists(td[0][0].mpoles.shape[0])
    lev_list = list(td.keys())
    for lv in lev_list[1:]:
        #won't need local expansions at root node,
        #won't use L2L at leaf nodes. 
        for k in td[lv].keys():
            #get local expansion from all interactions
            for i in td[lv][k].ilist:
                if i in td[lv].keys():
                    get_m2l(td[lv][k], td[lv][i], cl1, pl1)
            #get translated local expansion for all children,
            #unless you are a leaf node, in which case you have no children
            if lv != lev_list[-1]:
                for c in td[lv][k].children:
                    if c in td[lv+1].keys():
                        get_l2l(td[lv+1][c], td[lv][k], cl2, pl2)

downward_pass(tree_dict)

#EVALUTE FORCE/POTENTIAL
def get_far_force(charges, parray, lexp, c):
    l = np.arange(lexp.shape[0])
    for i in range(len(parray)):
        q = charges[parray['types'][i]]
        z = parray['x'][i] - c[0]
        z = z + 1j*(parray['y'][i]-c[1])
        f = np.sum(l*lexp*np.power(z, l-1))/(2*np.pi)
        parray['fx'][i] = -1*q*np.real(f)
        parray['fy'][i] = q*np.imag(f)
        
        
def get_loc_force(charges, parray):
    for i in range(len(parray)):
        for j in range(i+1, len(parray)):
            qi = charges[parray['types'][i]]
            qj = charges[parray['types'][j]]
            dxij = parray['x'][i] - parray['x'][j]
            dyij = parray['y'][i] - parray['y'][j]
            rij = np.sqrt((dxij**2) + (dyij**2))
            fij = qi*qj/(2*np.pi*(rij**2))
            parray['fx'][i] += fij*dxij
            parray['fy'][i] += fij*dyij
            parray['fx'][j] -= fij*dxij
            parray['fy'][j] -= fij*dyij
            
            
def get_neigh_force(charges, lparray, nparray):
    for i in range(len(lparray)):
        for j in range(len(nparray)):
            qi = charges[lparray['types'][i]]
            qj = charges[nparray['types'][j]]
            dxij = lparray['x'][i] - nparray['x'][j]
            dyij = lparray['y'][i] - nparray['y'][j]
            rij = np.sqrt((dxij**2) + (dyij**2))
            fij = qi*qj/(2*np.pi*(rij**2))
            lparray['fx'][i] += fij*dxij
            lparray['fy'][i] += fij*dyij
            nparray['fx'][j] -= fij*dxij
            nparray['fy'][j] -= fij*dyij            
        
        
def get_tot_force(charges, td):
    l = len(td.keys()) - 1
    for k in td[l].keys():
        get_far_force(charges, td[l][k].parray, td[l][k].lexp, td[l][k].center)
        get_loc_force(charges, td[l][k].parray)
        for n in td[l][k].neighs:
            if n < k:
                if n in td[l].keys():
                    get_neigh_force(charges, td[l][k].parray, td[l][n].parray)

get_tot_force(charges, tree_dict)

def lib_fmm2d(particles, n, p, c):
    #holds all nodes at each level
    tree_dict = {}
    for l in range(n):
        tree_dict[l] = {}
    #liberal in sense that every node stores its center, children, parents, and neighbors.
    root = LibFmm2dNode(tree_dict, particles.parray, n, p)
    set_leaf_mpoles2d(tree_dict, particles.charges)
    upward_pass(tree_dict)
    downward_pass(tree_dict)
    get_tot_force(particles.charges, tree_dict)