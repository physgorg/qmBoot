QMBoot Numerical Bootstrap Package

As seen in https://arxiv.org/abs/2209.14332

by George Hulsey

Mar 2023

Dependencies: 

	(Python)
	sympy 
	numpy
	tqdm
	matplotlib
	subprocess
	multiprocessing

	(Other)
	GNU multiple-precision arithmetic (GMP)
	SDPA-GMP

##########################################################

# OVERVIEW OF FILES

##########################################################

-- sym_recursion.py

This file contains code which computes the moment recursion for a given potential on some specified domain. 

Problems may be specified on the real line or on the real half line with Robin boundary conditions at zero. The boundary conditions are handled by including the anomaly terms and expressing the anomalies in terms of moments. 

The main use is to return the matrices F_n(E) such that M = F_0(E) + \sum F_n(E)x_n >= 0. They are returned as one-argument lambda functions which in turn return K x K matrices. These matrices along with their block decomposition information are ready to input into SDPA-GMP, after the proper formatting. 

An example usage of the 'momentSeq' function is:
	
	potential = "x^2+x^4"

	coeffs = getPolyCoeffs(potential)

	K = 4

	domain = 'R'

	moments,variables = momentSeq(2*K-1,coeffs,domain,BC = 'Dirichlet',v = True)

The option "v" specifies verbosity, which has the program output lots of messages to assist in debugging. 

This file should be imported in general, and its functions are applied in the other python files included. 

##########################################################

-- qmboot_lib.py

This file contains classes and search functions. This is essentially the home of higher-level executable code. 

The class "RationalSpectrumProblem" is designed to take in an energy array and solve the bootstrap SDP for the whole array. It can handle rational potentials on R or the half line R+ as it stands. 

Use of this file requires the hard coding of two directories: the directory of the script and a "data" folder and the path to the SDPA-GMP executable. These are specified in the top lines of the file. 

Example: 

	path_to_SDPA_GMP = "/Users/gorg/sdpa/sdpa-gmp-7.1.3/" 

	working_directory = "/Users/gorg/qmBootstrap" 

	output_data_directory = "/Users/gorg/qmBootstrap/data" 


##########################################################

-- qmboot_funcs.py

This file currently (July 2023) only contains one function "Krange_Search", which takes as input a potential and searches an energy range for eigenvalues over a range of depths. This eventually returns a dictionary of results. 

In any future development, higher-level functions can be placed here. 

