# import libraries
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sb


def PCA(normalized_csv_file, n_components):
    # get covariance matrix where 
    ## the diagonal of the matrix is the normal variance 
    ### other entries show distance between mean and point (x,y)
    #### and all other points
    cov_matrix = np.cov(normalized_csv_file, rowvar = False)
    #print(cov_mat)
    
    # Eigen vector are perpendicular of the best fit line of the data
    ## Eigen values are distance between point and best fit
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    
    # sort the eigen values descendingly indicated by their indexes
    sorted_index = np.argsort(eigen_values)
    sorted_index = np.flip(sorted_index)
    
    # sort the eigen values descendigly according to the index that 
    ## was obtained previously
    sorted_eigenvalue = eigen_values[sorted_index]
    
    # sort eigen vector according to their sorted eigen values
    ## uses sorted index to get their correct sort
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    
    # get n_components (2 in this case) biggest vectors
    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
    
    # calculate and get the reduced data required
    csv_file_reduced = np.dot(eigenvector_subset.transpose(), normalized_csv_file.transpose()).transpose()
    return csv_file_reduced

# Import csv file
csv_file = pd.read_csv(r'E:\University\Semester 7\Analysis and Design of Algorithms (CSEN 707)\group3.csv', delimiter=',', header=None, skiprows=1, names=["Height","Weight","BMI","L_Sh","L_Arm"])
#data.head()
#print(csv_file)

# get data mean
csv_file_mean = csv_file[["Height", "Weight", "BMI", "L_Sh", "L_Arm"]].mean()
#print(csv_file_mean)

# get data standard deviation
csv_file_std = csv_file[["Height", "Weight", "BMI", "L_Sh", "L_Arm"]].std()
#print(csv_file_std)

# get data variance
csv_file_var = csv_file[["Height", "Weight", "BMI", "L_Sh", "L_Arm"]].var()
#print(csv_file_var)

# normalizing the data
data = ((csv_file - csv_file_mean) / csv_file_std).abs()
#print(data)

# reduce data to r values
r = 1000
data = data.iloc[:r]

# Apply PCA function to data
csv_file_reduced = PCA(data, 2)

# Creating a Pandas DataFrame of reduced Dataset and naming the 
## n_components (2 in this case) biggest value to x & y
PCA_csv_file = pd.DataFrame(csv_file_reduced, columns=['X', 'Y'])

# Plotting of the scatterplot
plt.figure(figsize=(5, 5))
sb.scatterplot(data=PCA_csv_file, x='X', y='Y')
data_reduced = PCA_csv_file

# Change here to 3 in the following run
K = 5

# Select random observation as centers
Centers = (data_reduced.sample(n=K))
#print(Centers)

#Assign all the points to the closest cluster centroid
## Recompute centers of newly formed clusters
### Repeat step 3 and 4

diff = 1
j = 0

while (diff > 0.01):
    data_tmp = data_reduced
    i = 1
    for index1, row_c in Centers.iterrows():
        # for each x, get (center-x)^2  
        centers_x_distance = data_tmp['X'].map(lambda x: (x-row_c['X'])**2)
        # for each y, get (center-y)^2
        centeres_y_distance = data_tmp['Y'].map(lambda x: (x-row_c['Y'])**2)
        # sum of (center-x)^2 + (center-y)^2
        tmp_sum = centers_x_distance + centeres_y_distance
        # store distance between center and each point
        data_reduced[i] = np.sqrt(tmp_sum)
        i += 1

    C = []
    # choose closest center to each point according to value of K
    if(K == 5):
        data_reduced['Cluster'] = data_reduced[[1, 2, 3, 4, 5]].idxmin(axis=1)
    if(K == 3):
        data_reduced['Cluster'] = data_reduced[[1, 2, 3]].idxmin(axis=1)
    # grouby by cluster and get mean of X & Y for each center(cluster)
    Centers_new = data_reduced.groupby(["Cluster"]).mean()[["X", "Y"]]
    # Get the loop running until difference reaches its minimum or near 0.1
    if (j == 0):
        diff = 1
        j += 1
    else:
        diff = (Centers_new['X'] - Centers['X']).sum() + (Centers_new['Y'] - Centers['Y']).sum()
        diff = abs(diff)
       #print(diff.sum())
    # set old centers to new centers in order to compare them in 
    ## the next loop cycle 
    Centers = Centers_new
    color = ['black', 'green', 'blue', 'yellow', 'orange']

# filter clusters in order to color them
## draw each one of them separtely
for k in range(K):
    data = data_reduced[data_reduced["Cluster"] == k + 1]
    plt.scatter(data['X'], data['Y'], c=color[k])

# calculate percentage of each cluster
p = ((data_reduced['Cluster'].value_counts()/r) * 100)
# print percentage of each cluster
for i,j in enumerate(p):
    print(f'Percentage of Cluster {i+1} is {j}')

# plotting
plt.scatter(Centers['X'], Centers['Y'], c = 'red')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()