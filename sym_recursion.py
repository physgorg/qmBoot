# Symbolic bootstrap recursion

# part of QMBoot package

import sympy as sp
from sympy import Poly # symbolic algebra
import numpy as np
from tqdm import tqdm

verbose = False # for debugging, mostly

#########################################################################

# setup functions

def getPolyCoeffs(Vstr): # translate polynomial into list of coeffs. Assume variable is string 'x'
	xx = sp.Symbol('x')
	ls = {'x':xx}
	vv = sp.sympify(Vstr,locals = ls,convert_xor = True) # interprets 'x^2' as 'x**2'
	vp = Poly(vv,xx,domain = 'RR')
	aa = list(vp.all_coeffs())
	aa.reverse()
	return aa

def createPrimals(n): # returns a dict of symbolic primals x1,x2,..,xn for sympy manipulation
	string = ['x'+str(d) for d in range(1,n+1)]
	xv = sp.symbols(string)
	return xv

def spHankelize(arr): # turn (odd length) array into a Hankel matrix (as a sympy dtype)
	K = int((len(arr)+1)/2)
	return sp.Matrix(K,K,lambda i,j: arr[i+j])

#########################################################################

# recursion on moments

def RRmomentSeq(N,Vcoeffs,mass = 1/2, v = False):

	pcoeffs = Vcoeffs
	deg_p = len(pcoeffs)-1 # degree of polynomial potential
	
	undetermined_moments = deg_p - 1
	
	if v: print("\nDefining {} unknown moment variables...".format(undetermined_moments))

	e = sp.Symbol('e')
	xs = createPrimals(undetermined_moments)
	variables = [e] + xs
	X = [1] + xs
	if v: print('\nvariables:',variables)

	# moment x_{p} is lowest determined
	avec = np.asarray(pcoeffs) # polynomial coeffs

	if N < len(X): # undetermined moments (primals)
		return X[:N],xs[:N]

	for j in tqdm(range(len(X),N+1)): # recursion determines all higher moments
		n = j + 1 - deg_p
		if n == 1:
			const = (2*deg_p + 4*n)*avec[deg_p]
			t1 = 4*n*e*X[n-1]
			ts = [(2*m + 4*n)*avec[m]*X[n+m-1] for m in range(deg_p)]

			newterm = 1/const*(t1 - sum(ts))
			X = X + [newterm]
		elif n == 2:
			const = (2*deg_p + 4*n)*avec[deg_p]
			t1 = 4*n*e*X[n-1]
			ts = [(2*m + 4*n)*avec[m]*X[n+m-1] for m in range(deg_p)]

			newterm = 1/const*(t1 - sum(ts))
			X = X + [newterm]

		elif n >= 3:
			const = (2*deg_p + 4*n)*avec[deg_p]
			t1 = 4*n*e*X[n-1]
			t2 = 1/(2*mass)*n*(n-1)*(n-2)*X[n-3]
			ts = [(2*m + 4*n)*avec[m]*X[n+m-1] for m in range(deg_p)]

			newterm = 1/const*(t1 + t2 - sum(ts))
			X = X + [newterm]

		# there are three cases here as a lazy way of avoiding an IndexError

	return X,variables


def RpmomentSeq(N,Vcoeffs,BC, L = 0, mass = 1/2,v = True):

	pcoeffs = Vcoeffs
	deg_p = len(pcoeffs)-1 # degree 

	undetermined_moments = deg_p - 1
	avec = np.asarray(pcoeffs) # polynomial coeffs

	if v: print("Defining {} undetermined moments...".format(undetermined_moments))

	e = sp.Symbol('e')
	xs = createPrimals(undetermined_moments)
	X = [1] + xs
	
	variables = [e] + xs
	
	# define extra variables for anomalies
	if BC == "Dirichlet" or BC == 0:
		# here, a == 0
		psiL2 = 0
		
		psi_primeLpsiL = 0 

		if v: print("Boundary conditions at L = {}: a = 0; Dirichlet".format(L))
	elif BC == "Neumann" or BC == 'oo':
		# here, a = 'oo'

		# This integrates the (n = 0) depth recursion and determines psi0^2 in terms of moments
		psiL2 = 1/(e-avec[0])*sum([m*avec[m]*X[m-1] for m in range(1,deg_p+1)])
		# print('pl2',psiL2)
		psi_primeLpsiL = 0 # zero by Neumann

		psi_primeL2 = 0 # zero by Neumann

		if v: print("Boundary conditions at L = {}: Neumann (a = oo)".format(L))

	else:
		try:
			a = float(BC)

			# This integrates the (n = 0) depth recursion and determines psi0^2 in terms of moments
			psiL2 = 1/(2*(e-avec[0]) + 1/(mass*a**2))*sum([2*m*avec[m]*X[m-1] for m in range(1,deg_p+1)])
			psi_primeLpsiL = -1*psiL2/a
			psi_primeL2 = psiL2/a**2
			if v: print("Boundary conditions: a = -\u03C8'({})/\u03C8({}) = {}".format(L,L,BC))

		except ValueError:
			print("Invalid boundary condition. Try 'Dirichlet','Neumann', or a real number.")
			return None

	def anom_fprime(n): # anomaly A(f'_n) = <(H^* - H)f'_n(x)>
		if L != 0:
			print("L != 0 not supported. Change coords so boundary is at zero.")
			return None
		else:
			if n == 2:
				return -1/(mass)*psiL2
			else:
				return 0

	def anom_fmom(n): # anomaly 2iA(f_n p) = 2i<(H^* - H)f_n(x)p>

		# FOR THE TIME BEING, THIS DOES NOT WORK FOR L =/= 0

		if L == 0:
			if n == 0:
				# print("EVALUATED THE KEY ANOMALY")
				return 1/mass*psi_primeL2 + 2*psiL2*(e - pcoeffs[0])
			elif n == 1:
				return -1/mass*psi_primeLpsiL
			elif n > 1:
				return 0
		else:
			print("L != 0 not supported. Change coords so boundary is at zero.")
			return None

	# moment x_{p} is lowest determined
	if N < len(X):
		return X[:N],xs[:N]
	for j in tqdm(range(len(X),N+1)):
		n = j + 1 - deg_p
		if n == 1:
			const = (2*deg_p + 4*n)*avec[deg_p]
			t1 = 4*n*e*X[n-1]
			ts = [(2*m + 4*n)*avec[m]*X[n+m-1] for m in range(deg_p)]

			newterm = sp.expand(1/const*(t1 - sum(ts) - anom_fprime(1) + anom_fmom(1)))
			X = X + [newterm]
		elif n == 2:
			const = (2*deg_p + 4*n)*avec[deg_p]
			t1 = 4*n*e*X[n-1]
			ts = [(2*m + 4*n)*avec[m]*X[n+m-1] for m in range(deg_p)]

			newterm = sp.expand(1/const*(t1 - sum(ts)- anom_fprime(2) + anom_fmom(2)))
			X = X + [newterm]

		elif n >= 3:
			const = (2*deg_p + 4*n)*avec[deg_p]
			t1 = 4*n*e*X[n-1]
			t2 = 1/(2*mass)*n*(n-1)*(n-2)*X[n-3]
			ts = [(2*m + 4*n)*avec[m]*X[n+m-1] for m in range(deg_p)]

			newterm = sp.expand(1/const*(t1 + t2 - sum(ts) - anom_fprime(n) + anom_fmom(n)))
			X = X + [newterm]

	return X,variables


def momentSeq(N,Vcoeffs,domain,L = 0,BC = None,v = False):

	mass = 1/2 # We use these conventions throughout. It may be altered here. 

	# option for polynomial potential
	# domain: either R, R+, or [a,b]

	if domain == "R" or domain == "(-oo,oo)":
		# real line problem
		if v: print("Initializing recursion on the real line.")

		return RRmomentSeq(N,Vcoeffs,mass = mass,v = v)

	elif domain == "R+": 
		# half line problem. 
		if BC == None:
			print("No boundary information supplied. Terminating")
			return None 
		elif L != 0:
			print("L != 0 is not supported right now. change yo coordinates.")
			return None
		else:

			if v: print("Initializing recursion on a half line.")

			return RpmomentSeq(N,Vcoeffs,BC,L = L,mass = mass,v = v)

	elif domain[1] == '[' and domain[-1] == "]":

		return None
		# interval problem. THIS IS TABLED FOR THE MOMENT. 

def getFmatFuncs(K,Vcoeffs,domain,BC = None,L = 0,v = True):

	# return Fmats as function of energy 'e'. They are returned as one-argument lambda funcs

	if domain == "R" or domain == "(-oo,oo)":

		n_moments = 2*K - 2

		moments, variables = momentSeq(n_moments,Vcoeffs,'R',v = v) # symbolic recursion happens here

		# here we will have variables = [e,x1,x2,...,xn]

		if v: print("\nProblem has {} total undetermined variables, excluding SDP slack vars.\n".format(len(variables[1:])))

		moments = sp.Array(moments)

		blockInfo = [(K,K)] # block structure of constraint mats. always symmetric blocks for us
		e = variables[0] # energy variable
		xs = variables[1:] # unknown moments & linear variables

		# get the constant matrix
		if v: print("Computing matrix F0...")
		const_portion = moments.subs([(x,0) for x in xs])
		F0 = sp.lambdify(e,spHankelize(const_portion)) # it returns as a lambda function object
		Fmats = [F0]
		
		moms_homog = moments - const_portion 

		if v: print("Computing constraint matrices...")
		for x in tqdm(xs):
			subslist = [(xx,0) if xx != x else (x,1) for xx in xs ]
			portion = moms_homog.subs(subslist)
			Fmats = Fmats + [sp.lambdify(e,spHankelize(portion))]

		return blockInfo, [Fmats] # list of one-variable lambdas

	elif domain == "R+" or domain[-3:] == "oo)":

		n_moments = 2*K - 1

		moments, variables = momentSeq(n_moments,Vcoeffs,'R+', BC = BC,L = L,v = v) # symbolic recursion happens here

		if v: print("Problem has {} undetermined variables.".format(len(variables[1:])))

		moments = sp.Array(moments)

		blockInfo = [(K,K),(K,K)]

		e = variables[0] # energy variable
		xs = variables[1:] # unknown moments & linear variables

		momLists = [moments[:-1],moments[1:]] # for constructing Hamburger, Stieltjes matrices

		allMats = []

		for momList in momLists:
		
			const_portion = momList.subs([(x,0) for x in xs])
			F0 = sp.lambdify(e,spHankelize(const_portion)) # it returns as a lambda function object
			Fmats = [F0]
			
			moms_homog = momList - const_portion
			for x in xs:
				subslist = [(xx,0) if xx != x else (x,1) for xx in xs]
				portion = moms_homog.subs(subslist)
				Fmats = Fmats + [sp.lambdify(e,spHankelize(portion))]

			allMats.append(Fmats)

		return blockInfo, allMats # list of one-variable lambdas

	
#########################################################################

if __name__ == '__main__':

	potential = "x^2+x^4"

	coeffs = getPolyCoeffs(potential)

	K = 4

	domain = 'R'

	moments,variables = momentSeq(2*K-1,coeffs,domain,BC = 'Dirichlet',v = True)

	for mom in moments:
		print(mom)

	




