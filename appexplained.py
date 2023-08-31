# Imports
# ==============================================================================
# Standard libraries for data manipulation, visualization, and file operations
import pandas as pd
import trimesh
import matplotlib.pyplot as plt 
from PIL import Image
import streamlit as st
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from matplotlib import cm
import uuid
from conversion import convert_to_stl
from dotenv import load_dotenv
import multiprocessing
from psycopg2 import pool
from itertools import islice
from multiprocessing import Manager

load_dotenv()
np.random.seed(42)

# Database connection details
# ==============================================================================
DB_HOST = "db.hkarwpsgcnhxngaybxdv.supabase.co"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")

# Create a connection pool for the database
connection_pool = pool.SimpleConnectionPool(
    1, 20,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

# Dictionary to map abbreviations to product families
abbr_dict = {
    'SM': 'sleeve_bearing',
    'FM': 'sleeve_bearing_with_flange',
    'TM': 'thrust_washer',
    'FL': 'flange_bearing_iglidur',
    'XUM': 'liner_drylin',
    'PRM': 'piston_ring',
    'RLM': 'knife_edge_roller',
    'SRM': 'lead_screw_nut',
    'MCM': 'clip_bearing',
    'MCI': 'clip_bearing'
}


# Utility Functions
# ==============================================================================

''' Function to extract the product family from a filename based on abbreviations '''
def extract_product_family_from_filename(filename):
    for abbr, product_family in abbr_dict.items():
        if abbr in filename:
            return product_family
    return None


''' Function to transform point cloud to its canonical form.
    Canonical form ensures the point cloud is centered at the origin and oriented 
    in a consistent manner. This makes comparisons between point clouds more meaningful.
'''
def get_canonical_form(point_cloud):
    # Calculate the centroid of the point cloud
    centroid = point_cloud.mean(axis=0)
    
    # Subtract the centroid coordinates from all points, moving the centroid to the origin
    point_cloud = point_cloud - centroid
    
    # Compute the scatter matrix H
    H = np.dot(point_cloud.T, point_cloud) / point_cloud.shape[0]
    
    # Perform eigen decomposition to get the orientation
    w, u = np.linalg.eigh(H)
    u = u[:,[2, 1, 0]]
    w = w[[2, 1, 0]]
    
    # Correct the orientation if necessary
    if np.linalg.det(u) < 0:
        u[:,0] *= -1
        
    # Align the point cloud to the canonical axes
    point_cloud = np.dot(point_cloud, u)
    return point_cloud


''' Function to rotate a point cloud by a specified angle around the z-axis.
    This can be useful for comparing point clouds from different orientations.
'''
def rotate_point_cloud(point_cloud, angle):
    # Convert the angle to radians
    angle = np.deg2rad(angle)
    
    # Define the rotation matrix for a rotation around the z-axis
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    
    # Apply the rotation to the point cloud
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix)
    return rotated_point_cloud


''' Function to save the uploaded file locally.
    It handles both STL and STP file formats.
'''
def save_uploadedfile(uploadedfile):
    # Determine the local path where the file will be saved
    filepath = os.path.join("tempDir", uploadedfile.name)
    
    # If the uploaded file is a STP file, convert it to STL format
    if uploadedfile.name.split('.')[-1].lower() == 'stp':
        stl_bytes = convert_to_stl(uploadedfile)
        filepath = filepath.replace('.stp', '.stl')

    # Write the file (or converted file) to local storage
    with open(filepath, "wb") as f:
        if uploadedfile.name.split('.')[-1].lower() == 'stp':
            f.write(stl_bytes)
        else:
            f.write(uploadedfile.getbuffer())
    
    # Notify the user that the file has been saved
    st.success(f"Saved File:{filepath} in tempDir")
    return filepath


''' Function to preprocess the uploaded file.
    This involves sampling the 3D model to produce a point cloud, 
    and then transforming that point cloud to its canonical form.
'''
def preprocess_file(file_path):
    # Load the 3D model from the file
    mesh = trimesh.load(file_path)
    
    # Sample the 3D model to produce a point cloud
    point_cloud = mesh.sample(5000)
    
    # Transform the point cloud to its canonical form
    point_cloud = get_canonical_form(point_cloud)
    return point_cloud


''' Function to compute the centroid of a 3D mesh.
    The centroid provides a single point that represents the average position 
    of all points in the mesh. It's useful for alignment and comparison tasks.
'''
def compute_centroid_from_mesh(file_path):
    # Load the 3D model from the file
    mesh = trimesh.load(file_path)
    
    # Return the centroid of the mesh
    return mesh.centroid


''' Function to render a 2D projection of a 3D point cloud.
    This visualization aids in understanding the spatial arrangement of the point cloud.
'''
def render_2d_projection(file_path, file_name):
    unique_id = uuid.uuid4()
    
    # Preprocess the file to get the point cloud
    point_cloud = preprocess_file(file_path)
    
    # Define colors based on the z-coordinates for visualization
    colors = cm.viridis((point_cloud[:, 2] - point_cloud[:, 2].min()) / (point_cloud[:, 2].max() - point_cloud[:, 2].min()))
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], color=colors)
    
    # Save the visualization to a temporary image file
    img_filename = f'temp_{unique_id}.png' 
    plt.savefig(img_filename)
    
    # Display the image in the Streamlit app
    image = Image.open(img_filename)
    st.image(image, caption=f'2D Projection of {file_name}', use_column_width=True)
    
    # Clean up: remove the temporary image file
    if os.path.exists(img_filename): 
        os.remove(img_filename)


''' Function to preselect CAD files from the database based on centroid proximity.
    It retrieves the nearest neighbors based on the centroids of their 3D models.
'''
def preselect_files_by_centroid(centroid_coords):
    # Get a database connection from the pool
    connection = connect_to_database()
    cursor = connection.cursor()
    
    # Convert the centroid coordinates to a well-known text (WKT) string representation
    centroid_wkt = f"POINTZ({centroid_coords[0]} {centroid_coords[1]} {centroid_coords[2]})"
    
    # SQL query to select files based on centroid proximity
    query = """
    SELECT 
        f.filename,
        ST_Distance(f.centroid, ST_GeomFromText(%s, 4326)) AS centroid_distance
    FROM 
        "11iguscad" f
    ORDER BY 
        centroid_distance ASC
    LIMIT 1000;
    """
    cursor.execute(query, (centroid_wkt,))
    results = cursor.fetchall()
    
    # Close the database connection
    connection.close()
    
    return results


''' Function to execute a detailed comparison between the uploaded file and the preselected files.
    This comparison uses the Hausdorff distance metric to quantify the similarity 
    between two point clouds.
'''
def final_comparison(target_point_cloud, refined_files):
    results = []

    # Fetch the point clouds of the refined files from the database
    stored_point_clouds = get_point_cloud_from_db(refined_files)

    # Compute the Hausdorff distance for each point cloud
    for filename, stored_point_cloud in stored_point_clouds.items():
        distance = max(directed_hausdorff(target_point_cloud, stored_point_cloud)[0],
                       directed_hausdorff(stored_point_cloud, target_point_cloud)[0])
        results.append((filename, distance))

    # Sort the results by distance and return the top 10 closest matches
    return [filename for filename, _ in sorted(results, key=lambda x: x[1])][:10]


''' Function to download a blob from Azure blob storage.
    This function downloads a specific blob (file) from a given container 
    in Azure blob storage to a specified destination on the local file system.
'''
def download_blob(blob_service_client, container_name, blob_name, dest_folder):
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    blob_container_client = blob_service_client.get_container_client(container_name)
    
    # Create a list of all blobs within the container
    blobs_list = blob_container_client.list_blobs()
    
    # Download the desired blob
    with open(os.path.join(dest_folder, blob_name), "wb") as my_blob:
        download_stream = blob_client.download_blob()
        my_blob.write(download_stream.readall())


''' Function to get a database connection from the connection pool.
    This function retrieves an active connection from the connection pool 
    which can be used for executing database queries.
'''
def connect_to_database():
    return connection_pool.getconn()


''' Function to release a database connection back to the connection pool.
    After executing queries, it's important to release the connection so that 
    it can be reused by other parts of the application or by other users.
'''
def release_connection(connection):
    connection_pool.putconn(connection)


''' Function to rotate a point cloud around the x-axis by a specified angle.
    This can be useful for comparing point clouds from different orientations.
'''
def rotate_point_cloud_around_x(point_cloud, angle):
    angle = np.deg2rad(angle)
    
    # Define the rotation matrix for a rotation around the x-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    
    # Apply the rotation to the point cloud
    return np.dot(point_cloud, rotation_matrix)


''' Function to rotate a point cloud around the z-axis by a specified angle.
    This is another utility for handling different orientations.
'''
def rotate_point_cloud_around_z(point_cloud, angle):
    angle = np.deg2rad(angle)
    
    # Define the rotation matrix for a rotation around the z-axis
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    # Apply the rotation to the point cloud
    return np.dot(point_cloud, rotation_matrix)


''' Function to download a batch of files and compare them against the uploaded point cloud.
    This function handles both the downloading and the comparison of each file in the batch.
'''
def download_and_compare(args):
    uploaded_point_cloud, filenames_batch, progress_messages = args
    results = []

    # Download the files in the batch
    for filename in filenames_batch:
        stored_point_cloud = get_point_cloud_from_db(filename)
        max_distance = 0
        best_distance = float("inf")
        best_angle = 0

        # Compare the uploaded point cloud with various rotations of the stored point cloud
        for angle in [0, 90, 180, 270]:
            rotated_point_cloud = rotate_point_cloud(stored_point_cloud, angle)
            distance = max(directed_hausdorff(uploaded_point_cloud, rotated_point_cloud)[0],
                           directed_hausdorff(rotated_point_cloud, uploaded_point_cloud)[0])
            if distance < best_distance:
                best_distance = distance
                best_angle = angle
            max_distance = max(distance, max_distance)

        product_family = extract_product_family_from_filename(filename)
        results.append((filename, (best_distance, best_angle, product_family)))

    return results


''' Function to yield batches of a specified size from an iterable.
    This is a utility function that helps in breaking down a larger list into smaller, more manageable chunks.
'''
def batch(iterable, batch_size):
    iterable = iter(iterable)
    while True:
        chunk = tuple(islice(iterable, batch_size))
        if not chunk:
            return
        yield chunk


''' Function to compare the uploaded point cloud against a preselected set of files.
    This comparison is parallelized for efficiency using multiprocessing.
'''
def compare_files(uploaded_point_cloud, selected_files):
    np.random.seed(42) 
    manager = Manager()
    progress_messages = manager.list()
    
    # Break the selected files into batches for parallel processing
    batch_size = 10  # This number can be adjusted based on preferences and system capabilities
    batches = list(batch(selected_files, batch_size))

    # Parallelize the comparison using multiprocessing
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(download_and_compare, [(uploaded_point_cloud, filenames_batch, progress_messages) for filenames_batch in batches])
    
    # Flatten the results to a single list
    results = [item for sublist in results for item in sublist]

    # Sort and return the top matches based on the computed distances
    max_distance = max([distance for _, (distance, _, _) in results])
    results = sorted(results, key=lambda x: x[1][0])[:10]
    top_matches = [[distance, {"filename": filename, "product_family": product_family}, 1 - distance / max_distance] for filename, (distance, angle, product_family) in results]

    return top_matches


''' Function to execute a detailed comparison of a single file against the uploaded point cloud.
    This function handles the detailed comparison for a specific file, considering different rotations.
'''
def compare_single_file(args):
    uploaded_point_cloud, filename, stored_point_cloud, idx, total, progress_messages = args
    
    product_family = extract_product_family_from_filename(filename)
    family_string = f"({product_family})" if product_family else ""
    progress_messages.append(f"{family_string} Comparing file {idx}/{total}: {filename}")

    best_distance = float("inf")
    best_angle_z = 0
    best_angle_x = 0

    # Compare the uploaded point cloud with various rotations of the stored point cloud
    for angle_z in [0, 90]:
        rotated_point_cloud_z = rotate_point_cloud_around_z(stored_point_cloud, angle_z)
        for angle_x in [0, 90]:
            final_rotated_point_cloud = rotate_point_cloud_around_x(rotated_point_cloud_z, angle_x)
            distance = max(directed_hausdorff(uploaded_point_cloud, final_rotated_point_cloud)[0],
                           directed_hausdorff(final_rotated_point_cloud, uploaded_point_cloud)[0])
            if distance < best_distance:
                best_distance = distance
                best_angle_z = angle_z
                best_angle_x = angle_x

    return filename, (best_distance, (best_angle_z, best_angle_x), product_family)


''' Function to refine the comparison between the uploaded point cloud and a single file in the preselected set.
    This function focuses on refining the results by using the Hausdorff distance metric to compare 
    point clouds after accounting for potential rotations.
'''
def refine_single_file(args):
    uploaded_point_cloud, filename, stored_point_cloud = args
    best_distance = float("inf")
    
    # Compare the uploaded point cloud with two rotations of the stored point cloud
    for angle in [0, 90]:
        rotated_point_cloud = rotate_point_cloud(stored_point_cloud, angle)
        distance = max(directed_hausdorff(uploaded_point_cloud, rotated_point_cloud)[0],
                       directed_hausdorff(rotated_point_cloud, uploaded_point_cloud)[0])
        best_distance = min(best_distance, distance)
        
    return filename, best_distance


''' Function to fetch point cloud data from the database for multiple files.
    This function retrieves stored point cloud data based on a specified resolution.
'''
def get_point_cloud_from_db(filenames, resolution='100'):
    # Get a database connection from the pool
    connection = connect_to_database()
    cursor = connection.cursor()

    # Adjust the columns queried based on the desired resolution
    if resolution == '100':
        query_columns = 'point_cloud_100_x, point_cloud_100_y, point_cloud_100_z'
    elif resolution == '5000':
        query_columns = 'point_cloud_5000_x, point_cloud_5000_y, point_cloud_5000_z'
    else:
        raise ValueError("Invalid resolution value. Expected '100' or '5000'.")
    
    # Formulate SQL query to retrieve the necessary columns
    query = f"""
    SELECT filename, {query_columns}
    FROM "11iguscad"
    WHERE filename = ANY(%s);
    """
    cursor.execute(query, (filenames,))
    data = cursor.fetchall()

    # Release the database connection
    cursor.close()
    release_connection(connection)

    return {row[0]: np.column_stack(row[1:]) for row in data}


''' Function to classify the product family of a set of filenames.
    This function uses a majority voting mechanism among the given filenames to determine 
    the most likely product family.
'''
def classify_product_family(filenames):
    families = [extract_product_family_from_filename(filename) for filename in filenames]
    most_common_family = max(set(families), key=families.count)
    return most_common_family


''' Function to execute a refined comparison using the Hausdorff distance metric on the preselected files.
    This function narrows down the selection from the preselected files by performing a detailed comparison.
'''
def refine_with_hausdorff(uploaded_point_cloud, preselected_files):
    # Fetch the point clouds of the preselected files from the database
    stored_point_clouds = get_point_cloud_from_db(preselected_files)

    # Use multiprocessing to compute the Hausdorff distance for each point cloud in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(refine_single_file, [(uploaded_point_cloud, filename, stored_point_clouds[filename]) for filename in preselected_files])
        
    return sorted(results, key=lambda x: x[1])


''' Function to compute the centroid of a 3D mesh.
    The centroid represents the geometric center of the object.
'''
def compute_centroid_from_mesh(file_path):
    # Load the 3D mesh from the file
    mesh = trimesh.load(file_path)
    
    # Return the computed centroid of the mesh
    return mesh.centroid


# Main Function
# ==============================================================================
''' Main function to handle the CAD Matching API logic.
    It covers the entire process from uploading a file to visualizing the top matches.
'''
def main():
    # Display a title for the Streamlit app
    st.title('CAD Matching API')
    
    # Allow the user to upload a file for comparison
    uploaded_file = st.file_uploader("Choose a file for comparison", type=['stl', 'stp'], key='comparison_upload')

    # If the user uploads a file
    if uploaded_file is not None:
        st.text("Saving uploaded file...")
        
        # Save the uploaded file to a local directory
        comp_filepath = save_uploadedfile(uploaded_file)
        
        # Load the 3D model from the saved file
        mesh = trimesh.load(comp_filepath)
        
        st.text("Preprocessing uploaded file...")
        # Preprocess the uploaded file to obtain its canonical point cloud
        point_cloud_5000 = preprocess_file(comp_filepath) 
        # Compute the centroid of the 3D model
        centroid = compute_centroid_from_mesh(comp_filepath)

        # Sample a smaller point cloud from the 3D model for quick comparisons
        point_cloud_100 = mesh.sample(100)  
        point_cloud_100 = get_canonical_form(point_cloud_100)

        st.text("Preselecting files based on centroids...")
        
        # Display the progress messages of the preselection in an expandable section
        with st.expander('Preselection Progress Messages'):
            # Preselect files from the database based on their centroids
            preselected_results = preselect_files_by_centroid(tuple(centroid))
            
            # If files were preselected
            if preselected_results:
                # Compute a score for each preselected file based on centroid distances
                max_centroid_distance = max([result[1] for result in preselected_results])
                df_preselected = pd.DataFrame(preselected_results, columns=["Filename", "Centroid Distance"])
                df_preselected["Product Family"] = df_preselected["Filename"].apply(extract_product_family_from_filename).replace({None: 'NA'})
                df_preselected["Score (%)"] = df_preselected['Centroid Distance'].apply(lambda x: round((1 - x / max_centroid_distance) * 100, 1))
                # Display a table of the preselected files
                st.table(df_preselected[["Filename", "Product Family", "Score (%)"]])
                
                # Extract the filenames from the preselected results
                selected_files = [result[0] for result in preselected_results]
            else:
                st.warning("No files were preselected based on centroids!")
        
        # Display the top 10 matches from the preselection
        st.write("Top 10 Matches from preselection:")
        st.table(df_preselected.head(10)[["Filename", "Product Family", "Score (%)"]])

        # Extract the filenames from the first 5 rows of the preselected results
        top_5_preselected = df_preselected['Filename'].head(5).tolist()
        # Classify the product family based on majority voting
        preselected_family = classify_product_family(top_5_preselected)
        # Display the classified product family in bold
        st.markdown(f"Classified Product Family: **{preselected_family}**")

        # Extract filenames from the preselected results
        selected_files = [result[0] for result in preselected_results]
        
        st.text("Fetching Point Clouds (Wird noch schneller werden)...")
        # Fetch the point clouds for the selected files from the database
        all_point_clouds = get_point_cloud_from_db(selected_files, resolution='5000')
        st.text("Comparing files...")
        
        # Manager for multiprocessing to handle progress messages
        manager = Manager()
        progress_messages = manager.list()

        # Display the comparison progress messages in an expandable section
        with st.expander('Comparison Progress Messages'):
            np.random.seed(42)
            # Use multiprocessing to compare the uploaded file against the selected files
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(compare_single_file, [(point_cloud_5000, filename, all_point_clouds[filename], i+1, len(selected_files), progress_messages) for i, filename in enumerate(selected_files)])
                
            # Convert the results into a dataframe for easy visualization
            df_comparison = pd.DataFrame(results, columns=["Filename", "Comparison Data"])
            df_comparison["Product Family"] = df_comparison["Filename"].apply(extract_product_family_from_filename).replace({None: 'NA'})
            df_comparison["Score (%)"] = df_comparison["Comparison Data"].apply(lambda x: round((1 - x[0] / max([data[0] for _, data in results])) * 100, 1))
            df_comparison["Hausdorff Distance"] = df_comparison["Comparison Data"].apply(lambda x: x[0])
            df_comparison = df_comparison.sort_values(by="Score (%)", ascending=False)
            # Display a table of the comparison results
            st.table(df_comparison[["Filename", "Product Family", "Hausdorff Distance", "Score (%)"]])

        # If there were any comparison results
        if results:
            # Compute the top matches based on the Hausdorff distance
            max_distance = max([distance for _, (distance, _, _) in results])
            results = sorted(results, key=lambda x: x[1][0])[:10]
            top_matches = [[distance, {"filename": filename, "product_family": product_family}, 1 - distance / max_distance] for filename, (distance, angle, product_family) in results]

            # Convert the top matches into a dataframe for easy visualization
            df_matches = pd.DataFrame(
                {
                    "Filename": [match[1]['filename'] for match in top_matches],
                    "Product Family": [match[1]['product_family'] if match[1]['product_family'] else 'NA' for match in top_matches],
                    "Score (%)": [round(match[2]*100, 1) for match in top_matches]
                }
            )
            # Display the top 10 matches from the refined comparison
            st.write("Top 10 Matches from refined comparison:")
            st.table(df_matches)

            # Extract the filenames from the first 5 rows of the comparison results
            top_5_matches = df_matches['Filename'].head(5).tolist()
            # Classify the product family based on majority voting
            matches_family = classify_product_family(top_5_matches)
            # Display the classified product family in bold
            st.markdown(f"Classified Product Family: **{matches_family}**")

            # Create a client to interact with Azure Blob Storage
            blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
            filenames_for_dropdown = [match[1]["filename"] for match in top_matches]

            # Display visualizations of the top 3 matches
            st.write("Top 3 Matches visualized:")
            col1, col2, col3 = st.columns(3)
            for i, match in enumerate(top_matches[:3]):
                with col1 if i == 0 else col2 if i == 1 else col3:
                    # Download the 3D model file for the match from Azure Blob Storage
                    download_blob(blob_service_client, 'bearings', match[1]["filename"], 'tempDir')
                    # Render a 2D projection of the 3D model and display it
                    render_2d_projection(os.path.join("tempDir", match[1]["filename"]), match[1]["filename"])
            
            # Display a visualization of the uploaded file
            st.write("Your uploaded File:")
            render_2d_projection(comp_filepath, uploaded_file.name)
        else:
            st.write("No matches found!")

# Execute the main function when the script is run
if __name__ == '__main__':        
    main()