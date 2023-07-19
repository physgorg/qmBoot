# QMBoot Classes & assorted funcs

#########################################################################

import numpy as np 
import os
import subprocess # interface with shell
from tqdm import tqdm # progress bar
import multiprocessing as mp # parallelization of SDPA-GMP evaluation
import matplotlib.pyplot as plt

# local code import
from sym_recursion import *

parallelize = True # use multiprocessing parallelization procedure

path_to_SDPA_GMP = "/Users/gorg/sdpa/sdpa-gmp-7.1.3/" # executable sdpa-gmp should live in this directory

working_directory = "/Users/gorg/qmBootstrap" # directory of data files, python files, etc
										 # this should/could be set automatically using 'os' library. 
										 # I ran into issues doing so, so I just hard coded. 
										 # The problem is that SDPA wants to be run from its directory

output_data_directory = "/Users/gorg/qmBootstrap/data" # directory for data to be placed in and read from

#########################################################################

# General Function Library

def SDPAformat(arr,extraspace = False): # turn a numpy array into SDPA text input format, to write in text file
	try:
		x = float(arr)
		return "{{"+str(x)+"}}"
	except TypeError:
		rows = ['{' + ', '.join(str(entry) for entry in row) + '}' for row in arr]
		if extraspace:
			return '{\n' + ',\n'.join(rows) + '\n}'
		else:
			return '{' + ', '.join(rows) + '}'

def slist2list(string): # convert stringed list to list of floats
	string = string[:-1]
	string = string.split(',')
	return [float(x) for x in string]

def SDPA_Fmats(Fv,no_t = False): # create Fmats with SDPA format: -F0,-Id,F1,F2,...
	F0 = np.array(-1*Fv[0],dtype = np.double)
	Fv = Fv[1:]
	Fv = [np.array(x,dtype = np.double) for x in Fv]
	if not no_t:
		return [F0] + [-1*np.identity(F0.shape[0])] + Fv
	elif no_t:
		return [F0] + [np.zeros(F0.shape)] + Fv

def getIntervals(x,y,eps = 0): # gets x intervals where yvals > -1*eps. Cutoff eps available but not currently used. 
	# get intervals of positivity
	try:
		pos_inds = [i for i in range(len(x)) if y[i] > -1*eps]
	except IndexError:
		pos_inds = [i for i in range(len(y)) if y[i] > -1*eps] 
	cons = consecutive(pos_inds)
	intervals = []
	for inds in cons: # take care with boundaries
		if len(inds) == 0:
			return None
		elif inds[0]>0:
			minx = x[inds[0]-1]
		else:
			minx = x[inds[0]]
		try:
			maxx = x[inds[-1]+1]
		except IndexError:
			maxx = x[inds[-1]]
		intervals.append([minx,maxx])
	return intervals

def printKrangeResults(result_dict,all = True): # prints results in a nicely formatted manner

	if all:
		kys = list(result_dict.keys())
	else:
		kys = list(result_dict.keys())[-1:]

	for k in kys:

		intervals = result_dict[k]

		print("Results at K = {}; {} intervals found".format(k,len(intervals)))

		for interval in intervals:
			midpt = (interval[0]+interval[1])/2
			
			uncert = abs(interval[0]-midpt)

			print("\t{} \u00b1 {}".format(midpt,uncert))

def consecutive(data, stepsize=1): 
    return np.split(data, np.where(np.diff(data) > stepsize+1)[0]+1)
    # this function now will now neglect a 1-step gap
		
#########################################################################

# Parallelized Functions (these must exist in the top-level scope)

def doSDPA_GMP(fname_pair,keep_infiles = True): # call SDPA. 

	# get numerical data, create input and output files
	infile,outfile = fname_pair
	
	os.chdir(path_to_SDPA_GMP) # change the global path variable above

	# run sdpa-gmp from the command line, redirecting output
	os.system("./sdpa_gmp" + " " + infile + " " + outfile + " > /dev/null")

	os.system('rm '+infile)

def grepObjVec(outfile_name): # use some bash to extract the line of interest from the SDPA output file

	eind = outfile_name.index("e=")
	eind2 = outfile_name.index("_K")
	energyString = outfile_name[eind+2:eind2]

	objvec = subprocess.check_output("grep -A1 'xVec' "+outfile_name+" |  grep '{' | sed -e 's/{/"+energyString+",/'  -e 's/}.*//' -e 's:+::g'", shell=True, text=True)

	val = slist2list(objvec)

	return val

def grepObjVecTonly(outfile_name): # use some bash to extract the line of interest from the SDPA output file

	eind = outfile_name.index("e=")
	eind2 = outfile_name.index("_K")
	energyString = outfile_name[eind+2:eind2]

	objvec = subprocess.check_output("grep -A1 'xVec' "+outfile_name+" |  grep '{' | sed -e 's/{/"+energyString+",/'  -e 's/}.*//' -e 's:+::g'", shell=True, text=True)

	val = slist2list(objvec)

	return val[1]

#########################################################################

class RationalSpectrumProblem:

	def __init__(self,pfunc,domain,K,BC = None,L = 0,verbose = False,readvals = False, Vlabel = None):

		# if "readvals = True", then code will attempt to read results of SDPA if they are already written. Convenient for plotting etc.

		# Vlabel is a string labeling directories of IO files

		# directories
		self.home_directory = working_directory # directory of this script
		self.data_directory = output_data_directory  # might need to be modified if data directory name changes
		# keep/delete SDPA-GMP IO files
		self.keep_infiles = False # if False, deletes all SDPA-GMP input files
		self.keep_outfiles = False #if False, deletes all SDPA-GMP output files

		if verbose:
			print("Potential is V(x) = {} on domain {}".format(pfunc,domain))
		
		# globalize
		self.readvals = readvals # bool; if true, tries to read off values printed to file instead of recomputing
		self.K = K
		if domain == 'R':
			self.truncation = 2*K-2
		elif domain == "R+":
			self.truncation = 2*K-1
		self.verbose = verbose
		self.dom = domain
		self.L = L 
		self.bc = BC
		self.V = pfunc

		if verbose: print("\nDefining problem at depth K = {}; computing {} moments.\n".format(K,self.truncation))

		########################

		self.Vcoeffs = getPolyCoeffs(pfunc)

		self.degP = len(self.Vcoeffs)-1

		# determine number of primal variables
		self.n_primals = self.degP # including t, dp  - 1 undetermined moments
		if verbose: print("primals are: (t,x1,...x{})".format(self.n_primals-1))
		
		# directory management
		if Vlabel == None:
			allcfs = np.asarray(self.Vcoeffs)
			self.Vlabel = ''.join(['_'+str(x) for x in allcfs.astype(np.float16)])
		
		self.createDatadir()

		if verbose: print("Data is in directory {}\n".format(self.datadir))
		# this block does the recursion and ends up with lambda'd 
		
		# skip this if readvals
		if not self.readvals:
			
			recursion_result = getFmatFuncs(K,self.Vcoeffs,domain,BC = self.bc,v = verbose)

			if verbose: print("Recursion completed.")

			bs, Fs = recursion_result

			self.blockInfo = bs 
			self.FFuncs = Fs

		self.outfiles = []

	def createDatadir(self): # creates a directory to store sdpa input/output files
		os.chdir(self.data_directory)
		dirname = "K="+str(self.K)+"_BC-"+str(self.bc)+"_V"+self.Vlabel
		os.system('mkdir '+dirname)
		self.datadir = self.data_directory + "/" + dirname
		os.chdir(os.path.dirname(os.path.abspath(__file__)))

	def createSDPAFiles(self,energy): # create SDPA IO files for a specific energy

		# define filenames
		out_fname = self.datadir+"/e="+str(energy)+'_K='+str(self.K)+"_D="+str(self.dom)+'.out'
		inp_fname = self.datadir+"/e="+str(energy)+'_K='+str(self.K)+"_D="+str(self.dom)+'.dat'

		# create an output file
		os.system("touch "+out_fname)
		self.outfiles.append(out_fname)

		# create a string to print to file
		tofile = "*SDP Scan Problem: V = "+str(self.V)+"\n* K = "+str(self.K)+"; eval = "+str(energy)
		if self.bc == "Dirichlet" or self.bc == "Neumann":
			tofile += "; Domain = {} with BCs: {}".format(self.dom,self.bc)
		else:
			tofile += "; Domain = {} with BCs: a = {}".format(self.dom,self.bc)
		tofile += "\n"+str(self.n_primals)+" = mDIM"+"\n"

		# dictate the block-diag structure of the matrices
		if self.dom == "R":

			# just one block: Hamburger block
			tofile += "1 = nBLOCK"+"\n"+str(self.K)+" = bLOCKsTRUCT"+"\n"

			# write the cost function
			c = str([-1] + [0 for i in range(self.n_primals-1)]) # cost vector

			cstr = "{"+c[1:-1]+"}"
			tofile += cstr+"\n"

			# evalute the lambda'd Fmats
			evaluated = [np.asarray(f(energy)) for f in self.FFuncs[0]]

			# get them in SDP format: -F0, -Id, F1, F2, ... Fm, so that M = \sum Fn x_n - F0 - tId >= 0
			matrices = SDPA_Fmats(evaluated)	

			# format them as strings to print to file
			for matrix in matrices:
				stringed = SDPAformat(matrix,extraspace = True)
				tofile += stringed + "\n"

			# create & write input file
			f = open(inp_fname,'w')
			f.write(tofile)
			f.close()

		if self.dom == "R+" and (self.bc == "Dirichlet" or self.bc == 0): # Dirichlet BC case

			# here have blockInfo [(K,K),(K,K)]
			# two blocks; one Hamburger matrix and one Stieltjes matrix
			tofile += "2 = nBLOCK"+"\n"+"({},{})".format(self.K,self.K)+" = bLOCKsTRUCT"+"\n"

			# write the cost function
			c = str([-1] + [0 for i in range(self.n_primals-1)]) # cost vector
			cstr = "{"+c[1:-1]+"}"
			tofile += cstr+"\n"

			# get the two block componennts of each Fmat
			eval1 = SDPA_Fmats([np.asarray(f(energy)) for f in self.FFuncs[-2]])
			eval2 = SDPA_Fmats([np.asarray(f(energy)) for f in self.FFuncs[-1]])
			
			# add components in blocks
			for i in range(len(eval1)):
			 	addStr = "{\n"+"{}\n{}".format(SDPAformat(eval1[i]),SDPAformat(eval2[i])) + "\n}\n"
			 	tofile += addStr

			f = open(inp_fname,'w')
			f.write(tofile)
			f.close()

		elif self.dom == 'R+': # Neumann, Robin BC case

			tofile += "2 = nBLOCK"+"\n"+"({},{})".format(self.K,self.K)+" = bLOCKsTRUCT"+"\n"

			# write the cost function
			c = str([-1] + [0 for i in range(self.n_primals-1)]) # cost vector
			cstr = "{"+c[1:-1]+"}"
			tofile += cstr+"\n"

			# print(self.FFuncs)

			psiL_epsilon = 0 # this constraints psi(L)^2 > epsilon for some epsilon > 0. 

			# get the block componennts of each Fmat
			eval1 = SDPA_Fmats([np.asarray(f(energy)) for f in self.FFuncs[0]])
			eval2 = SDPA_Fmats([np.asarray(f(energy)) for f in self.FFuncs[1]])
			# the SDPA_Fmats wrapper gets it to form : -F0 -t*Id + \sum(x_n F_n) >= 0, which is standard for SDPA

			for i in range(len(eval1)):
			 	addStr = "{\n"+"{}\n{}".format(SDPAformat(eval1[i]),SDPAformat(eval2[i])) + "\n}\n"
			 	tofile += addStr

			f = open(inp_fname,'w')
			f.write(tofile)
			f.close()

		return inp_fname,out_fname

	def rangeSDPsolve(self,erange): # solve all SDPs for the files just created

		if self.readvals: # if just reading values, don't re-solve
			pass
		else:

			if self.verbose:
				print("Creating SDPA input and output files...")

			# create all SDPA files
			fnames = [self.createSDPAFiles(energy) for energy in tqdm(erange)]

			if self.verbose:
				print('Sending data to SDPA-GMP...')

			if len(erange) < 5 or not parallelize: # if small datasize don't bother parallelizing

				for pair in fnames:
					doSDPA_GMP(pair,keep_infiles = self.keep_infiles)
					if self.keep_infiles:
						print('BONES KEEPING INFILES')

					if not self.keep_infiles:
						os.system('rm '+pair[0])

			elif parallelize: # otherwise, use multiprocessing parallelization

				# parallelized evaluation of sdpa-gmp
				pool = mp.Pool(mp.cpu_count())

				# use imap function to allow tqdm progress bar
				outfiles = list(tqdm(pool.imap(doSDPA_GMP,fnames),total = len(erange)))

				pool.close()

			os.chdir(self.home_directory) # return to directory of script

			if self.verbose:
				print('SDPA-GMP completed.')

	def getResult(self,erange,t_only = True): # read results from file 

		res = []

		if self.verbose: print("Reading results from file...")

		for energy in tqdm(erange):

			outfile = self.datadir+"/e="+str(energy)+'_K='+str(self.K)+"_D="+str(self.dom)+'.out'

			val = grepObjVec(outfile)
			
			if not self.keep_outfiles: # if we want to get rid of the outfiles, 
				os.system('rm '+outfile)

			if not t_only: # if we are interested the value of the entire objective function,
				res.append(val)
			else: # otherwise return only the value of t, which should be > 0 for energy to be allowed
				res.append(val[1])

		return res 

			
	def getVals(self,erange,fname = 'vals.txt',t_only = True):

		if self.verbose:
			print('Grabbing values from SDPA output...')

		valfile = self.datadir+"/"+fname

		try: # open "vals.txt" file if it exists
			if self.readvals: # 
				f = open(valfile,'r')
				lines = f.readlines()
				f.close()
				vals = []
				for x in lines:
					x = x.split(',')
					x = [float(xx) for xx in x]
					vals.append(x)
				return np.asarray(vals)
			else:
				raise FileNotFoundError

		except FileNotFoundError:

			vals = []

			if self.verbose:
				print("No vals file found. Reading vals and creating file...")

			vals = self.getResult(erange,t_only = t_only)

			f = open(valfile,'w')

			for entry in vals:

				f.write(str(entry)+"\n")

			f.close()

			return np.asarray(vals)

if __name__ == '__main__':
	
	Ptest = "x^2+x^4"

	dom = 'R+'
	bc = 'Neumann'

	depths = [4,5]
	erange = np.linspace(0.1,5,200)


	for depth in depths:
		prob = RationalSpectrumProblem(Ptest,dom,depth,BC = bc,verbose = True,readvals =False)

		prob.rangeSDPsolve(erange)

		vals = np.asarray(prob.getVals(erange))
		print(vals)
		# print("\n",bc,"\n")
		plt.plot(erange,np.log(np.abs(vals)),label = "K = {}".format(depth))
		plt.xlabel("energy E")
		plt.ylabel("log(|t|) objective function")
		# plt.title("neumann BCs on half line")
				
	plt.legend()
	plt.show()

	



