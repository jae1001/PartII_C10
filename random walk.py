import matplotlib as mpl
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

M = 500      # Number of chains to average over (the higher the better)
N = 10000   # Number of monomer segments in each chain (>100 for random walk model to apply)
num_chains_to_visualize = 10

r_distances = []  # List to store the final distances

mpl.rcParams['legend.fontsize'] = 8

fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(projection='3d')

# Select 10 random chains to visualize
selected_chains = random.sample(range(M), num_chains_to_visualize)

for i in range(M):
    xyz = []
    cur = [0, 0, 0]

    for _ in range(N):
        axis = random.randrange(0, 3)
        cur[axis] += random.choice([-1, 1])
        xyz.append(cur[:])

    x, y, z = zip(*xyz)
    
    if i in selected_chains:
        ax.plot(x, y, z, label='Random walk '+str(i+1))
        ax.scatter(0, 0, 0, c='b', marker='x')             # Start point
        ax.scatter(x[-1], y[-1], z[-1], c='b', marker='o') # End point
    
    r_distance = math.sqrt(math.pow(x[-1],2) + math.pow(y[-1],2) + math.pow(z[-1],2))
    r_distances.append(r_distance)

# Calculate averages
r_avg = sum(r_distances) / M
r2_avg = sum(math.pow(d, 2) for d in r_distances) / M

# Create a histogram
plt.figure()
plt.hist(r_distances, bins=20, edgecolor='black')
plt.axvline(x=math.sqrt(N), color='red', linestyle='dashed', linewidth=2, label='Theoretical RMS')  # Theoretical value
plt.title('Histogram of Root Mean Square Distances')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.legend()
plt.show()
