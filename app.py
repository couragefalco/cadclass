# Imports
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

np.random.seed(42) 

# Load environment variables
load_dotenv()

# Database connection details
DB_HOST = "db.hkarwpsgcnhxngaybxdv.supabase.co"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")

# Create a connection pool
connection_pool = pool.SimpleConnectionPool(
    1, 20,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

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

def extract_product_family_from_filename(filename):
    for abbr, product_family in abbr_dict.items():
        if abbr in filename:
            return product_family
    return None

def get_canonical_form(point_cloud):
    centroid = point_cloud.mean(axis=0)
    point_cloud = point_cloud - centroid
    H = np.dot(point_cloud.T, point_cloud) / point_cloud.shape[0]
    w, u = np.linalg.eigh(H)
    u = u[:,[2, 1, 0]]
    w = w[[2, 1, 0]]
    if np.linalg.det(u) < 0:
        u[:,0] *= -1
    point_cloud = np.dot(point_cloud, u)
    return point_cloud

def rotate_point_cloud(point_cloud, angle):
    angle = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])  # rotation around z-axis
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix)
    return rotated_point_cloud

def save_uploadedfile(uploadedfile):
    filepath = os.path.join("tempDir", uploadedfile.name)
    if uploadedfile.name.split('.')[-1].lower() == 'stp':
        stl_bytes = convert_to_stl(uploadedfile)
        filepath = filepath.replace('.stp', '.stl')

    with open(filepath, "wb") as f:
        if uploadedfile.name.split('.')[-1].lower() == 'stp':
            f.write(stl_bytes)
        else:
            f.write(uploadedfile.getbuffer())
    st.success(f"Saved File:{filepath} in tempDir")
    return filepath

def preprocess_file(file_path):
    mesh = trimesh.load(file_path)
    point_cloud = mesh.sample(5000) 
    point_cloud = get_canonical_form(point_cloud)
    return point_cloud

def compute_centroid_from_mesh(file_path):
    """Compute the centroid of a 3D mesh."""
    mesh = trimesh.load(file_path)
    return mesh.centroid

def render_2d_projection(file_path, file_name):
    unique_id = uuid.uuid4()
    point_cloud = preprocess_file(file_path)
    colors = cm.viridis((point_cloud[:, 2] - point_cloud[:, 2].min()) / (point_cloud[:, 2].max() - point_cloud[:, 2].min()))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], color=colors)
    img_filename = f'temp_{unique_id}.png' 
    plt.savefig(img_filename)
    image = Image.open(img_filename)
    st.image(image, caption=f'2D Projection of {file_name}', use_column_width=True)
    if os.path.exists(img_filename): 
        os.remove(img_filename) # clean up

def preselect_files_by_centroid(centroid_coords):
    """Retrieve nearest neighbors based on centroids."""
    connection = connect_to_database()
    cursor = connection.cursor()

    # Convert the centroid_coords to a WKT string
    centroid_wkt = f"POINTZ({centroid_coords[0]} {centroid_coords[1]} {centroid_coords[2]})"

    # Use the uploaded file's centroid directly in the query
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
    connection.close()
    print(f"Preselected {len(results)} files based on centroids.")
    return results

def final_comparison(target_point_cloud, refined_files):
    """Final comparison using the 5000er point cloud."""
    results = []

    # Fetch the point clouds of the refined files
    stored_point_clouds = get_point_cloud_from_db(refined_files)

    # Compute the Hausdorff distance for each point cloud
    for filename, stored_point_cloud in stored_point_clouds.items():
        distance = max(directed_hausdorff(target_point_cloud, stored_point_cloud)[0],
                       directed_hausdorff(stored_point_cloud, target_point_cloud)[0])
        results.append((filename, distance))

    return [filename for filename, _ in sorted(results, key=lambda x: x[1])][:10]

def download_blob(blob_service_client, container_name, blob_name, dest_folder):
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    blob_container_client = blob_service_client.get_container_client(container_name)
    blobs_list = blob_container_client.list_blobs()
    with open(os.path.join(dest_folder, blob_name), "wb") as my_blob:
        download_stream = blob_client.download_blob()
        my_blob.write(download_stream.readall())

def connect_to_database():
    """Get a connection from the pool."""
    return connection_pool.getconn()

def release_connection(connection):
    """Release a connection back to the pool."""
    connection_pool.putconn(connection)

def rotate_point_cloud_around_x(point_cloud, angle):
    angle = np.deg2rad(angle)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    return np.dot(point_cloud, rotation_matrix)

def rotate_point_cloud_around_z(point_cloud, angle):
    angle = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return np.dot(point_cloud, rotation_matrix)

def download_and_compare(args):
    uploaded_point_cloud, filenames_batch, progress_messages = args
    results = []

    # Download the files in the batch
    for filename in filenames_batch:
        stored_point_cloud = get_point_cloud_from_db(filename)
        max_distance = 0
        best_distance = float("inf")
        best_angle = 0

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

def batch(iterable, batch_size):
    """Yield batches of size batch_size from iterable."""
    iterable = iter(iterable)
    while True:
        chunk = tuple(islice(iterable, batch_size))
        if not chunk:
            return
        yield chunk

def compare_files(uploaded_point_cloud, selected_files):
    np.random.seed(42) 
    manager = Manager()
    progress_messages = manager.list()
    
    # Break selected files into batches
    batch_size = 10  # You can adjust this number based on your preferences and system capabilities
    batches = list(batch(selected_files, batch_size))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(download_and_compare, [(uploaded_point_cloud, filenames_batch, progress_messages) for filenames_batch in batches])
    
    # Flatten the results
    results = [item for sublist in results for item in sublist]

    for message in progress_messages:
        st.text(message)

    max_distance = max([distance for _, (distance, _, _) in results])
    results = sorted(results, key=lambda x: x[1][0])[:10]
    top_matches = [[distance, {"filename": filename, "product_family": product_family}, 1 - distance / max_distance] for filename, (distance, angle, product_family) in results]

    return top_matches

def refine_single_file(args):
    uploaded_point_cloud, filename, stored_point_cloud = args
    best_distance = float("inf")
    for angle in [0, 90]:
        rotated_point_cloud = rotate_point_cloud(stored_point_cloud, angle)
        distance = max(directed_hausdorff(uploaded_point_cloud, rotated_point_cloud)[0],
                       directed_hausdorff(rotated_point_cloud, uploaded_point_cloud)[0])
        best_distance = min(best_distance, distance)
    return filename, best_distance

def compare_single_file(args):
    uploaded_point_cloud, filename, stored_point_cloud, idx, total, progress_messages = args
    
    product_family = extract_product_family_from_filename(filename)
    family_string = f"({product_family})" if product_family else ""
    progress_messages.append(f"{family_string} Comparing file {idx}/{total}: {filename}")

    best_distance = float("inf")
    best_angle_z = 0
    best_angle_x = 0

    # Iterate over the possible rotations around the Z-axis
    for angle_z in [0, 90]:
        rotated_point_cloud_z = rotate_point_cloud_around_z(stored_point_cloud, angle_z)
        # Now iterate over the rotations around the X-axis
        for angle_x in [0, 90]:
            final_rotated_point_cloud = rotate_point_cloud_around_x(rotated_point_cloud_z, angle_x)
            distance = max(directed_hausdorff(uploaded_point_cloud, final_rotated_point_cloud)[0],
                           directed_hausdorff(final_rotated_point_cloud, uploaded_point_cloud)[0])
            if distance < best_distance:
                best_distance = distance
                best_angle_z = angle_z
                best_angle_x = angle_x

    return filename, (best_distance, (best_angle_z, best_angle_x), product_family)


def refine_with_hausdorff(uploaded_point_cloud, preselected_files):
    np.random.seed(42) 
    """Refine the preselected files using Hausdorff distance with multiprocessing."""
    # Fetch the point clouds of the preselected files
    stored_point_clouds = get_point_cloud_from_db(preselected_files)

    # Use multiprocessing to compute the Hausdorff distance for each point cloud
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(refine_single_file, [(uploaded_point_cloud, filename, stored_point_clouds[filename]) for filename in preselected_files])
        
    return sorted(results, key=lambda x: x[1])

def get_point_cloud_from_db(filenames, resolution='100'):
    """Retrieve point cloud data for multiple files from the 3iguscad table based on resolution."""
    connection = connect_to_database()
    cursor = connection.cursor()

    # Adjust the columns queried based on the resolution
    if resolution == '100':
        query_columns = 'point_cloud_100_x, point_cloud_100_y, point_cloud_100_z'
    elif resolution == '5000':
        query_columns = 'point_cloud_5000_x, point_cloud_5000_y, point_cloud_5000_z'
    else:
        raise ValueError("Invalid resolution value. Expected '100' or '5000'.")

    # Only select the required columns
    query = f"""
    SELECT filename, {query_columns}
    FROM "11iguscad"
    WHERE filename = ANY(%s);
    """
    cursor.execute(query, (filenames,))
    data = cursor.fetchall()

    # Always good practice to close the cursor when done
    cursor.close()
    release_connection(connection)

    return {row[0]: np.column_stack(row[1:]) for row in data}

def classify_product_family(filenames):
    """Classify the product family based on majority voting from the given filenames."""
    families = [extract_product_family_from_filename(filename) for filename in filenames]
    most_common_family = max(set(families), key=families.count)
    return most_common_family

def main():
    st.title('CAD Matching API')
    uploaded_file = st.file_uploader("Choose a file for comparison", type=['stl', 'stp'], key='comparison_upload')

    if uploaded_file is not None:
        st.text("Saving uploaded file...")
        comp_filepath = save_uploadedfile(uploaded_file)
        
        mesh = trimesh.load(comp_filepath)
        
        st.text("Preprocessing uploaded file...")
        point_cloud_5000 = preprocess_file(comp_filepath)  # This is for the final comparison
        centroid = compute_centroid_from_mesh(comp_filepath)

        point_cloud_100 = mesh.sample(100)  
        point_cloud_100 = get_canonical_form(point_cloud_100)

        st.text("Preselecting files based on centroids...")
        
        with st.expander('Preselection Progress Messages'):
            preselected_results = preselect_files_by_centroid(tuple(centroid))
            
            if preselected_results:
                max_centroid_distance = max([result[1] for result in preselected_results])
                df_preselected = pd.DataFrame(preselected_results, columns=["Filename", "Centroid Distance"])
                df_preselected["Product Family"] = df_preselected["Filename"].apply(extract_product_family_from_filename).replace({None: 'NA'})
                df_preselected["Score (%)"] = df_preselected['Centroid Distance'].apply(lambda x: round((1 - x / max_centroid_distance) * 100, 1))
                st.table(df_preselected[["Filename", "Product Family", "Score (%)"]])
                
                selected_files = [result[0] for result in preselected_results]
            else:
                st.warning("No files were preselected based on centroids!")
        
        st.write("Top 10 Matches from preselection:")
        st.table(df_preselected.head(10)[["Filename", "Product Family", "Score (%)"]])

        # Extract the filenames from the first 5 rows
        top_5_preselected = df_preselected['Filename'].head(5).tolist()
        # Classify the product family based on majority voting
        preselected_family = classify_product_family(top_5_preselected)
        # Display the classified product family in bold
        st.markdown(f"Classified Product Family: **{preselected_family}**")

        selected_files = [result[0] for result in preselected_results]
        
        st.text("Fetching Point Clouds (Wird noch schneller werden)...")
        all_point_clouds = get_point_cloud_from_db(selected_files, resolution='5000')
        st.text("Comparing files...")
        manager = Manager()
        progress_messages = manager.list()

        with st.expander('Comparison Progress Messages'):
            np.random.seed(42) 
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(compare_single_file, [(point_cloud_5000, filename, all_point_clouds[filename], i+1, len(selected_files), progress_messages) for i, filename in enumerate(selected_files)])
                
            df_comparison = pd.DataFrame(results, columns=["Filename", "Comparison Data"])
            df_comparison["Product Family"] = df_comparison["Filename"].apply(extract_product_family_from_filename).replace({None: 'NA'})
            df_comparison["Score (%)"] = df_comparison["Comparison Data"].apply(lambda x: round((1 - x[0] / max([data[0] for _, data in results])) * 100, 1))
            df_comparison["Hausdorff Distance"] = df_comparison["Comparison Data"].apply(lambda x: x[0])
            df_comparison = df_comparison.sort_values(by="Score (%)", ascending=False)
            st.table(df_comparison[["Filename", "Product Family", "Hausdorff Distance", "Score (%)"]])

        if results:
            max_distance = max([distance for _, (distance, _, _) in results])
            results = sorted(results, key=lambda x: x[1][0])[:10]
            top_matches = [[distance, {"filename": filename, "product_family": product_family}, 1 - distance / max_distance] for filename, (distance, angle, product_family) in results]

            df_matches = pd.DataFrame(
                {
                    "Filename": [match[1]['filename'] for match in top_matches],
                    "Product Family": [match[1]['product_family'] if match[1]['product_family'] else 'NA' for match in top_matches],
                    "Score (%)": [round(match[2]*100, 1) for match in top_matches]
                }
            )
            st.write("Top 10 Matches from refined comparison:")
            st.table(df_matches)

            # Extract the filenames from the first 5 rows
            top_5_matches = df_matches['Filename'].head(5).tolist()
            # Classify the product family based on majority voting
            matches_family = classify_product_family(top_5_matches)
            # Display the classified product family in bold
            st.markdown(f"Classified Product Family: **{matches_family}**")


            blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
            filenames_for_dropdown = [match[1]["filename"] for match in top_matches]

            st.write("Top 3 Matches visualized:")
            col1, col2, col3 = st.columns(3)
            for i, match in enumerate(top_matches[:3]):
                with col1 if i == 0 else col2 if i == 1 else col3:
                    download_blob(blob_service_client, 'bearings', match[1]["filename"], 'tempDir')
                    render_2d_projection(os.path.join("tempDir", match[1]["filename"]), match[1]["filename"])
            
            st.write("Your uploaded File:")
            render_2d_projection(comp_filepath, uploaded_file.name)
        else:
            st.write("No matches found!")

if __name__ == '__main__':        
    main()




