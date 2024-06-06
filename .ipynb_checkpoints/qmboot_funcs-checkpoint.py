# SDP Searching functions

import numpy as np 
import os
import subprocess # interface with shell
from tqdm import tqdm # progress bar
import multiprocessing as mp # parallelization of SDPA-GMP evaluation
import matplotlib.pyplot as plt
import scipy.special as sc

# local code import
from sym_recursion import *
from qmboot_lib import *

###########################################################################

# KrangeSearch is the most basic. Important arguments are a potential, energy range, and K range. 

# it discretizes the energy range and does a search at each value of K, recording the intervals of positivity in a dict.

def Krange_Search(pfunc,domain,krange,elims,BC = None,L = 0,bins = 300,v = False,readvals = False):

	kr = np.arange(krange[0],krange[1]+1) # construct K range

	espaces = [np.linspace(elims[0],elims[1],num = bins)] # construct energy range

	result_dict = {}

	if v: print("WOOHOO! It's time to bootstrap.\n")

	for k in kr:

		intervals = []

		# define a bootstrap problem object
		if v: print("\nConstructing a bootstrap problem object at depth K = {}.\n".format(k))
		the_boot = RationalSpectrumProblem(pfunc,domain,k,BC = BC, L = L, verbose = v,readvals = readvals)

		for j in range(len(espaces)):

			espace = espaces[j]

			if not readvals:
				the_boot.rangeSDPsolve(espace)

			values = the_boot.getVals(espace,fname = "vals"+str(j)+".txt",t_only = True)

			intss =  getIntervals(espace,values)

			if intss == None:
				if v: print("No positivity detected in this interval. Disregarding...")

			else:
				intervals += intss

		result_dict[k] = intervals

		if v: printKrangeResults(result_dict)

		nint = len(intervals)

		if v:
			print('Completed search at depth {}; found'.format(k),nint,'intervals.')

		espaces = []

		if nint == 0:
			if v: print("No positivity intervals found.")
			return result_dict
		else:

			for i in range(nint-1):

				emin,emax = intervals[i]

				lowlowbins = round(bins/10)

				espaces.append(np.linspace(emin,emax,lowlowbins))

			final_emin, final_emax = intervals[-1]

			if final_emax == elims[1]:

				lowbins = round(bins/5)
			else:
				lowbins = round(bins/10)

			espaces.append(np.linspace(final_emin,final_emax,lowbins))


	return result_dict




if __name__ == "__main__":

	v = "x^2 + x^4"

	dom = 'R+'
	bc = "Neumann"

	Klims = [5,12]
	elims = [0.1,15]


	results = Krange_Search(v,dom,Klims,elims,
			BC = bc, v = True,readvals = False,bins = 300)
		
	





















