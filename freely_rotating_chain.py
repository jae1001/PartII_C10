import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import statistics
from numba import jit

def check_segment_angles(x, y, z):
    angles = []
    for i in range(1, len(x) - 1):
        # Vector of the previous segment
        v1 = np.array([x[i] - x[i-1], y[i] - y[i-1], z[i] - z[i-1]])
        
        # Vector of the next segment
        v2 = np.array([x[i+1] - x[i], y[i+1] - y[i], z[i+1] - z[i]])
        
        # Normalize the vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Calculate the angle between the vectors
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        
        # Convert to degrees
        angle_deg = np.degrees(angle)
        angles.append(angle_deg)
    
    return angles

def check_dihedral_angles(x, y, z):
    dihedrals = []
    for i in range(3, len(x)):
        # Vectors between atoms
        b1 = np.array([x[i-2] - x[i-3], y[i-2] - y[i-3], z[i-2] - z[i-3]])
        b2 = np.array([x[i-1] - x[i-2], y[i-1] - y[i-2], z[i-1] - z[i-2]])
        b3 = np.array([x[i] - x[i-1], y[i] - y[i-1], z[i] - z[i-1]])

        # Normal vectors
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        # Normalize normal vectors
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)

        # Calculate dihedral angle
        x_val = np.dot(n1, n2)
        y_val = np.dot(np.cross(n1, b2/np.linalg.norm(b2)), n2)
        dihedral = np.degrees(np.arctan2(y_val, x_val))
        dihedrals.append(dihedral)

    return dihedrals

def generate_polymer_chain(N, a, valence_angle):
    # Initialize arrays to store x, y, and z coordinates of each segment
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    z = np.zeros(N + 1)

    for i in range(1, N + 1):
        # Add a new segment at a fixed valence angle and random dihedral angle relative to the z-axis
        x[i], y[i], z[i] = add_segment(x[i - 1], y[i - 1], z[i - 1], a, valence_angle)

        # Rotate the entire chain so that the newly added segment is pointing along the z-axis
        x, y, z = rotate_chain_to_z_axis(x, y, z, i)

        # Shift the origin to the beginning of the newly added segment
        x, y, z = shift_origin(x, y, z, -x[i], -y[i], -z[i])

    return x, y, z

def add_segment(x, y, z, a, valence_angle):
    # Add a new segment at a fixed valence angle and random dihedral angle relative to the z-axis
    dihedral_angle = np.random.uniform(0, 2 * np.pi)
    new_x = x + a * np.sin(valence_angle) * np.cos(dihedral_angle)
    new_y = y + a * np.sin(valence_angle) * np.sin(dihedral_angle)
    new_z = z + a * np.cos(valence_angle)
    #print(f"new_x {new_x}, new_y {new_y}, new_z {new_z}")
    return new_x, new_y, new_z

@jit(nopython=True)
def shift_origin(x, y, z, shift_x, shift_y, shift_z):
    # Shift the origin of the coordinates
    shifted_x = x + shift_x
    shifted_y = y + shift_y
    shifted_z = z + shift_z
    return shifted_x, shifted_y, shifted_z

def rotate_chain_to_z_axis(x, y, z, i):
    # Normalize the input vector
    norm = np.linalg.norm([x[i], y[i], z[i]])
    vector = np.array([x[i], y[i], z[i]]) / norm

    # Calculate the rotation axis
    rotation_axis = np.cross([0, 0, 1], vector)
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Calculate the rotation angle
    rotation_angle = np.arccos(np.dot([0, 0, 1], vector))

    # Create the rotation matrix
    rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, rotation_angle)

    # Combine the coordinates into a matrix
    coordinates_matrix = np.column_stack((x, y, z))

    # Apply the rotation matrix
    rotated_coordinates_matrix = np.dot(coordinates_matrix, rotation_matrix)

    # Extract the rotated coordinates
    rotated_x, rotated_y, rotated_z = rotated_coordinates_matrix.T
    return (rotated_x, rotated_y, rotated_z)

@jit(nopython=True)
def rotation_matrix_from_axis_angle(axis, angle):
    # Create a 3D rotation matrix from an axis and an angle
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    x, y, z = axis
    rotation_matrix = np.array([
        [t * x**2 + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y**2 + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z**2 + c]
    ])

    return rotation_matrix

def calculate_squared_end_to_end_distance(x, y, z):
    # Calculate the squared end-to-end distance
    squared_distance = (x[-1] - x[0])**2 + (y[-1] - y[0])**2 + (z[-1] - z[0])**2
    return squared_distance

def test_end_to_end_distance():
    # Parameters
    M = 2000   # Number of chains to average over (the higher the better)
    N = 1000    # Number of monomer segments in each chain (>100 for random walk model to apply)
    a = 1.0    # Length of each segment
    valence_angle = 70.5*np.pi/180  # Valence angle in radians

    squared_distances = []  # List to store the squared distances

    for i in range(M):
        # Generate polymer chain coordinates
        x, y, z = generate_polymer_chain(N, a, valence_angle)

        # Calculate the squared end-to-end distance
        squared_distance = calculate_squared_end_to_end_distance(x, y, z)
        squared_distances.append(squared_distance)
        
    # Calculate the root mean square of the end-to-end distances
    mean_squared_distance = statistics.mean(squared_distances)
    rms_distance = np.sqrt(mean_squared_distance)
    root_squared_distances = np.sqrt(squared_distances)
    print(f"Root-Mean-Square End-to-End Distance: {rms_distance}")

    # Expected end-to-end distance for a freely jointed chain
    expected_distance = np.sqrt(N) * a 
    print(f"Freely-jointed End-to-End Distance: {expected_distance}")

    # Expected end-to-end distance for a freely rotating chain
    expected_distance = np.sqrt(N) * a * np.sqrt((1+np.cos(valence_angle))/(1-np.cos(valence_angle)))
    print(f"Freely rotating End-to-End Distance: {expected_distance}")

    # Create a histogram
    plt.figure()
    plt.hist(root_squared_distances, bins=20, edgecolor='black')
    plt.axvline(expected_distance, color='red', linestyle='dashed', linewidth=2, label='Theoretical RMS')  # Theoretical value
    plt.title('Histogram of Root Mean Square Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_the_chain():
    # Parameters
    N = 10000  # Number of segments
    a = 1.0  # Length of each segment
    valence_angle = 70.5*np.pi/180  # Valence angle in radians

    squared_distances = []  # List to store the squared distances

    # Generate polymer chain coordinates
    x, y, z = generate_polymer_chain(N, a, valence_angle)

    # Calculate the squared end-to-end distance
    squared_distance = calculate_squared_end_to_end_distance(x, y, z)
    squared_distances.append(squared_distance)

    # Calculate the root mean square of the end-to-end distances
    mean_squared_distance = statistics.mean(squared_distances)
    rms_distance = np.sqrt(mean_squared_distance)
    root_squared_distances = np.sqrt(squared_distances)
    print(f"Root-Mean-Square End-to-End Distance: {rms_distance}")

    # Check angles of the chain
    angles = check_segment_angles(x, y, z)
    print(f"Angles between segments (first 10): {angles[:10]}")
    print(f"Mean angle: {np.mean(angles):.2f} degrees")
    print(f"Standard deviation of angles: {np.std(angles):.2f} degrees")

    # Check dihedral angles of chain
    dihedrals = check_dihedral_angles(x, y, z)
    print(f"Dihedral angles (first 10): {dihedrals[:10]}")
    print(f"Mean dihedral: {np.mean(dihedrals):.2f} degrees")
    print(f"Standard deviation of dihedrals: {np.std(dihedrals):.2f} degrees")

    # Plot histogram of angle deviations
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.hist(angles, bins=50, range=(-180, 180), edgecolor='black')
    plt.axvline(np.degrees(valence_angle), color='red', linestyle='dashed', linewidth=2, label='Expected Angle')
    plt.title('Histogram of Angles')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot histogram of dihedrals
    plt.subplot(122)
    plt.hist(dihedrals, bins=50, range=(-180, 180), edgecolor='black')
    plt.title('Histogram of Dihedral Angles')
    plt.xlabel('Dihedral Angle (degrees)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Plot the polymer chain
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Run the test
#test_end_to_end_distance()

# Plot the chain
plot_the_chain()

