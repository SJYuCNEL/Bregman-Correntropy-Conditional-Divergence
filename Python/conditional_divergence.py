#        <Bregman-Correntropy Conditional Divergence>
# 	  
#   File:     conditional_divergence.py
#   Authors:  <Shujian Yu (Shujian.Yu@neclab.eu)> 
#             <Ammar Shaker (Ammar.Shaker@neclab.eu)>
# 
# NEC Laboratories Europe GmbH, Copyright (c) <2020>, All rights reserved.  
# 
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#  
#        PROPRIETARY INFORMATION ---  
# 
# SOFTWARE LICENSE AGREEMENT
# 
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
# 
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
# 
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor. 
# 
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
# 
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
# 
# COPYRIGHT: The Software is owned by Licensor.  
# 
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
# 
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
# 
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
# 
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
# 
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
# 
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
# 
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.  
# 
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
# 
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
# 
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
# 
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.  
# 
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
# 
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
# 
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
# 
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
# 
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
# 
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
# 
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

####################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
from scipy.linalg import fractional_matrix_power
from scipy.linalg import logm
import scipy.io
from numpy.linalg import inv
from numpy.linalg import eig
from numpy import transpose as trans
from copy import deepcopy

################### Kernel functions

def RBFkernel(x1,x2,alpha=0.5):
    """ Computing the RBF kernel between two vectors x1 and x2.
    K(x1 ,x2 )=\exp (-\frac {\|x1-x2\|^2}{2alpha^2})

    ----------
    x1 : np.array
        the first sample
    x2 : np.array
        the second sample
    alpha: float
        the kernel width
        
    Returns
    -------
     K(x1,x2) : float
        the RBF kernel result
    """

    return np.exp(-np.sum(np.power(x1-x2,2))/(2*alpha**2))

def linearkernel(x1,x2):
    """ Computing the linear kernel between two vectors x1 and x2.
    K(x1 ,x2 )  = x1^T x2
    ----------
    x1 : np.array
        the first sample
    x2 : np.array
        the second sample    
        
    Returns
    -------
     K(x1,x2) : float
        the linear kernel result
    """
    return np.sum(np.dot(x1,x2))


################### Useful matrix functions
def is_pos_def(x):
    """ Checks if the matrix x is positive semidefinite by checking that all eigenvalues are >=0
    ----------
    x : np.array
        the matrix to be checked

    Returns
    -------
      : Boolean
        whether the matrix x is positive semidefinite or not
    """
    return np.all(np.linalg.eigvals(x) >= 0)

def min_eigvals(x):
    """ Returns the minimum eigenvalues of matrix x
    ----------
    x : np.array
        the matrix to be checked

    Returns
    -------
      : float
        the smallest eigenvalue of matrix x
    """    
    return min(np.linalg.eigvals(x))


def sample_Cov_Mat(dim):
    """ Returns a random positive semidefinite covariate matrix
    ----------
    dim : int
        the dimension of the required matrix

    Returns
    -------
    C : np.array
        a random positive semidefinite covariate matrix
    """        
    a = np.random.uniform(-1,1,size=(dim, dim))
    O, r = np.linalg.qr(a,mode='complete')
    p = np.random.uniform(-1,1,dim)
    p = np.sort(p)
    D = np.diag(np.square(p))
    C = np.matmul(np.matmul(O.transpose(),D),O)
    return C
################### von Neumann divergence functions

def von_Neumann_divergence(A,B):
    """ Computing the von Neumann divergence between two positive semidefinite matrices A and B
    D_{vN}(A||B) = Tr(A (log(A)-log(B))-A+B)
    ----------
    A : np.array
        the first array
    B : np.array
        the second array    
        
    Returns
    -------
      : float
        the von Neumann divergence
    """    
    return np.trace(A.dot(logm(A)-logm(B))-A+B)

def von_Neumann_divergence_Eff( A,B):
    """ Computing the von Neumann divergence between two positive semidefinite matrices A and B efficiently
    D_{vN}(A||B) = Tr(A (log(A)-log(B))-A+B)
    ----------
    A : np.array
        the first array
    B : np.array
        the second array    
        
    Returns
    -------
      : float
        the von Neumann divergence
    """    
    #Divergence = np.trace(np.dot(A, logm(A)) - np.dot(A, logm(B)) - A + B)
    Aeig_val, Aeig_vec = eig(A)
    Beig_val, Beig_vec = eig(B) 
    Aeig_val, Aeig_vec = abs(Aeig_val), (Aeig_vec)
    Beig_val, Beig_vec = abs(Beig_val), (Beig_vec)
    Aeig_val[Aeig_val<1e-10] = 0
    Beig_val[Beig_val<1e-10] = 0

    A_val_temp, B_val_temp = deepcopy(Aeig_val), deepcopy(Beig_val)
    A_val_temp[Aeig_val <= 0] = 1
    B_val_temp[Beig_val <= 0] = 1

    part1 = np.sum(Aeig_val * np.log2(A_val_temp) - Aeig_val + Beig_val) 

    lambda_log_theta = np.dot(Aeig_val.reshape(len(Aeig_val),1), np.log2(B_val_temp.reshape(1, len(B_val_temp))))
    part2 = (np.dot(Aeig_vec.T, Beig_vec) **2) * lambda_log_theta
    part2 = -np.sum(part2)
    Divergence = part1 + part2

    return Divergence

################### log det divergence
def log_det_divergence(A,B):
    """ Computing the logDet divergence between two positive semidefinite matrices A and B
    D_{\ell D}(A||B) = \Tr(B^{-1}A) + \log_2\frac{|B|}{|A|} - n,
    ----------
    A : np.array
        the first array
    B : np.array
        the second array    
        
    Returns
    -------
      : float
        the logDet divergence
    """    

    cross_term = np.trace(np.matmul(A,np.linalg.inv(B))) - np.log(np.linalg.det(np.matmul(A,np.linalg.inv(B)))) - A.shape[0]
    return cross_term 

def log_det_divergenceEigSort(A,B):
    """ Computing the logDet divergence between two positive semidefinite matrices A and B efficiently
    D_{\ell D}(A||B) = \Tr(B^{-1}A) + \log_2\frac{|B|}{|A|} - n,
    ----------
    A : np.array
        the first array
    B : np.array
        the second array    
        
    Returns
    -------
      : float
        the logDet divergence
    """    

    Aeig_val,Aeig_vec = eig(A)
    idx = Aeig_val.argsort()[::-1]   
    Aeig_val = Aeig_val[idx]
    Aeig_vec = Aeig_vec[:,idx]

    Beig_val,Beig_vec = eig(B)    
    idx = Beig_val.argsort()[::-1]   
    Beig_val = Beig_val[idx]
    Beig_vec = Beig_vec[:,idx]
    
    Aeig_val = abs(Aeig_val)
    Beig_val = abs(Beig_val)
    Aeig_val[Aeig_val<1e-10] = 0
    Beig_val[Beig_val<1e-10] = 0
    length= A.shape[0]
    cross_term = 0
    for i in range(length):
        for j in range(length):
            cross_term += (Aeig_vec[:,i].dot(Beig_vec[:,j])**2)* (Aeig_val[i]/Beig_val[j] if (Aeig_val[i]>0 and Beig_val[j]>0) else 1)
        cross_term -= (np.log(Aeig_val[i]/Beig_val[i]) if (Aeig_val[i]>0 and Beig_val[i]>0) else 1)

    return cross_term - length

################### Centered Correntropy divergence

def corrent_matrix(data,kernel_size):
    """ 
    data: np.array
        data of size n x d, n is number of sample, d is dimension
    kernel_size: float
        the kernel width
    -------
    data: np.array
        a d x d (symmetric) center correntropy matrix
    """    
    dim = data.shape[1]
    corren_matrix = np.zeros(shape=(dim,dim))
    for i in range(dim):
        for j in range(i+1):
            corren_matrix[i,j] = corren_matrix[j,i] = sample_center_correntropy(data[:,i],data[:,j],kernel_size)
    
    return corren_matrix

def sample_center_correntropy(x,y,kernel_size):
    """ Computing the center correntropy between two vectors x and y
    ----------
    x : np.array
        the first sample
    y : np.array
        the second sample
    kernel_size: float
        the kernel width
        
    Returns
    -------
      : float
        center correntropy between X and Y
    """    

    twosquaredSize = 2*kernel_size**2
    bias = 0
    for i in range(x.shape[0]):
        bias +=sum(np.exp(-(x[i]-y)**2/twosquaredSize))
        #for j in range(x.shape[0]):
        #    bias +=np.exp(-(x[i]-y[j])**2/twosquaredSize)
    bias = bias/x.shape[0]**2
    
    corren = (1/x.shape[0]) * sum(np.exp(-(x-y)**2/twosquaredSize)) -bias
    return corren
