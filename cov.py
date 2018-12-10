#Program For: checking and comparing distance preservation in different samples of data or based on chosen eigenVectors
#Author: Ashok Suthar
#Input: 0 for using IRIS data, or any other number for randomly generating 100x100 matrix.
import numpy as np
import pandas as pd
import random
import math
import scipy.spatial.distance as sp
from sklearn import random_projection

#checking distance measures in actual data and covariance of data.
def data_cov_dif(adjustedData):
	#finding covariance_matrix of mean adjusted data
	cov = np.cov(adjustedData, rowvar=False)

	#mean of the covariance matrix of the actual data
	cov_mean = np.mean(cov, axis = 0)
	
	#Standard deviation of covariance matrix data
	cov_SD = math.sqrt(np.sum(np.square(cov-cov_mean)))/(len(cov)-1)

	print("Standard Deviation in covariance data")
	print(cov_SD)

	print("pdist of points in covariance matrix data")
	print(sp.pdist(cov))
	print("\n\n")

#checking distance measures in actual and reduced data (using most significant 50 eigenValues)
def data_reduced_data_diff(adjustedData):
	cov = np.cov(adjustedData,rowvar=False)
	#calculating eigenvalues and eigen vectors and sorting them according to eigenValues
	eigenValues, eigenVectors = np.linalg.eig(cov)
	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]

	#choosing 50 most signification eigenValues and corresponding eigenvectors
	EigenVectors = eigenVectors[:,:int(len(adjustedData)/2)]
	#finding transpose
	tran_EigenVectors = EigenVectors.transpose()
	print(np.shape(tran_EigenVectors))
	#finding data transpose
	tran_data = adjustedData.transpose()
	print(np.shape(tran_data))

	#reduced data(np.real is used to turn imaginary numbers into real ones)
	reduced_data = (np.real(np.matmul(tran_EigenVectors, tran_data)))

	#calculating mean of reduced data
	reduced_data_mean = np.mean(reduced_data, axis=0)

	#Standard deviation of reduced data
	reduced_data_SD = math.sqrt(np.sum(np.square(reduced_data-reduced_data_mean)))/(len(reduced_data)-1)

	print("Standard Deviation in reduced data")
	print(reduced_data_SD)

	#printing pdist() of reduced data
	print("pdist of points in reduced data matrix data")
	print(sp.pdist(reduced_data))
	print("\n\n")

#taking samples of data and checking the distance measures with all eigen vectors and eigen values
def data_sample_data_diff(data): 
	#converting data to normal list to use random.sample() function
	data = data.tolist()
	for i in range(1,11):
		print("Taking "+str(i)+"th  sample")
		#taking 20% random samples of data from actual data
		sample_data = random.sample(data,int(20*(len(data)/100)))
		#converting sample_data back to numpy array
		sample_data = np.array(sample_data)
		
		sample_data_mean = np.mean(data, axis=0)
		adjustedData = sample_data-sample_data_mean
	
		#finding covariance matrix of adjusted sample data
		cov = np.cov(adjustedData,rowvar=False)
		#calculating eigenvalues and eigen vectors and sorting them according to eigenValues
		eigenValues, eigenVectors = np.linalg.eig(cov)
		idx = eigenValues.argsort()[::-1]   
		eigenValues = eigenValues[idx]
		eigenVectors = eigenVectors[:,idx]
		#choosing all eigenValues and corresponding eigenvectors
		eigenVectors = eigenVectors[:,:len(sample_data)]
		#finding transpose
		tran_EigenVectors = eigenVectors.transpose()
		#finding data transpose
		tran_data = adjustedData.transpose()
		print(np.shape(tran_EigenVectors))
		print(np.shape(tran_data))
		# data using all the eigen values and eigen vectors
		allData = (np.real(np.matmul(tran_EigenVectors, tran_data)))

		#calculating mean of generated data
		allData_mean = np.mean(allData, axis=0)
		print(allData_mean)

		#Standard deviation of reduced data
		allData_SD = math.sqrt(np.sum(np.square(allData-allData_mean)))/(len(allData)-1)
		print("Standard Deviation in "+str(i)+"th sampled Data data")
		print(allData_SD)

		#printing pdist() of reduced data
		print("pdist of points in "+str(i)+"th sampled matrix data")
		print(sp.pdist(allData))
		print()
	print("\n\n")

#checking distance measures in actual and projected data(reduced dimension) according to user given dimensions of target data. (randomProjections)
def data_user_proj_data_diff(data,targ_dim):
	#finding minimum dimension reduction possible using JL lemma, while preserving pairwise distances upto a given eps value.
	min_dim = random_projection.johnson_lindenstrauss_min_dim(100,eps=0.1)
	print("min dim suggested by JL lemma= "+str(min_dim))
	#creating transformer matrix to use for projecting the input data to target data. if O = IR. transformer is R here.
	transformer = random_projection.SparseRandomProjection(n_components=targ_dim)
	#transforming given "data"(input) to "projected_data"(output) by using "transformer" as random matrix R.
	projected_data = transformer.fit_transform(data)
	print("new data dimensions after projection according to user provided target data dimension: "+str(np.shape(projected_data)))
	#printing pdist() of projected data
	print("pdist of points in projected data as per user provided target data dimension")
	print(sp.pdist(projected_data))
	print()
	print("\n\n")

#checking distance measures in actual and projected data(reduced dimension) using target dimension value according to JL lemma . (randomProjections)
def data_JL_proj_data_diff(data):
	#creating transformer matrix to use for projecting the input data to target data. if O = IR. transformer is R here.
	transformer = random_projection.SparseRandomProjection()
	#transforming given "data"(input) to "projected_data"(output) by using "transformer" as random matrix R.
	projected_data = transformer.fit_transform(data)
	print("new data dimensions after projection according to user provided target data dimension: "+str(np.shape(projected_data)))
	#printing pdist() of projected data
	print("pdist of points in JL projected data")
	print(sp.pdist(projected_data))
	print()
	print("\n\n")

def generate_data(data_type):
	if data_type == 0:
		df = pd.read_csv('iris.csv',sep=",",header=None)
		print("Taking Iris.data(150x4) as input data")
		data = df.iloc[:, :4]
		#converting to numpy array
		data = np.array(data)
	else:
		print("Generating normalized random data(100x10000) as input data")
		data = np.random.normal(0,5,(100,10000))
	return data


if __name__ == "__main__":
	data_type = int(input("Press '0' to work with iris data set or anyother number to work with randomly generated data:"))
	#calling generate_data() for data to be generated/read.
	data = generate_data(data_type)
	#mean of actual data
	data_mean = np.mean(data, axis=0)
	#taking user input for dimensionality of target data wanted by user
	targ_dim = int(input("Enter dimensionality of target data wanted by user:"))
	#Standard deviation of actual data
	data_SD = math.sqrt(np.sum(np.square(data-data_mean)))/(len(data)-1)
	print("Standard Deviation in actual data")
	print(data_SD)
	
	#printing pdist of actual data
	print("pdist of points in actual data")
	print(sp.pdist(data))

	#mean adjusting actual data
	mean_adjusted_data = data-data_mean
	#calling functions
	#data_cov_dif(mean_adjusted_data)
	#data_reduced_data_diff(mean_adjusted_data)
	#data_sample_data_diff(data)
	data_user_proj_data_diff(data,targ_dim)
	data_JL_proj_data_diff(data)
	print("Done!")



