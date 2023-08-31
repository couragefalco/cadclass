import os
import numpy as np
from azure.storage.blob import BlobServiceClient
import io
import trimesh
import logging
import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql
np.random.seed(42)



# Initialize PostgreSQL connection using environment variables
load_dotenv()
DB_HOST = "db.hkarwpsgcnhxngaybxdv.supabase.co"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
logging.getLogger('azure').setLevel(logging.ERROR)

def get_canonical_form(point_cloud):
    centroid = point_cloud.mean(axis=0)
    point_cloud = point_cloud - centroid
    H = np.dot(point_cloud.T, point_cloud) / point_cloud.shape[0]
    w, u = np.linalg.eigh(H)
    u = u[:, [2, 1, 0]]
    w = w[[2, 1, 0]]
    if np.linalg.det(u) < 0:
        u[:, 0] *= -1
    point_cloud = np.dot(point_cloud, u)
    return point_cloud

def connect_to_db():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return conn



def chunks(data, size=50):
    for i in range(0, len(data), size):
        yield data[i:i+size]

def create_table_if_not_exists(conn, table):
    """Create the table if it doesn't already exist."""
    cur = conn.cursor()

    create_table_sql = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {table} (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            centroid GEOMETRY(PointZ, 4326),  -- Modified to PointZ
            point_cloud_100_x REAL[],
            point_cloud_100_y REAL[],
            point_cloud_100_z REAL[],
            point_cloud_5000_x REAL[],
            point_cloud_5000_y REAL[],
            point_cloud_5000_z REAL[]
        )
    """).format(table=sql.Identifier(table))

    cur.execute(create_table_sql)
    conn.commit()
    cur.close()



def process_blob(blob):
    blob_name = blob.name
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    blob_data = blob_client.download_blob().readall()
    mesh = trimesh.load_mesh(io.BytesIO(blob_data), file_type='stl')
    
    point_cloud_5000 = mesh.sample(5000)
    point_cloud_5000 = get_canonical_form(point_cloud_5000)

    point_cloud_100 = mesh.sample(100)
    point_cloud_100 = get_canonical_form(point_cloud_100)

    centroid = mesh.centroid

    return {
        'filename': blob_name,
        'centroid': centroid,
        'point_cloud_100_x': point_cloud_100[:, 0].tolist(),
        'point_cloud_100_y': point_cloud_100[:, 1].tolist(),
        'point_cloud_100_z': point_cloud_100[:, 2].tolist(),
        'point_cloud_5000_x': point_cloud_5000[:, 0].tolist(),
        'point_cloud_5000_y': point_cloud_5000[:, 1].tolist(),
        'point_cloud_5000_z': point_cloud_5000[:, 2].tolist()
    }

def upload_point_cloud_to_db(conn, table, data):
    cur = conn.cursor()

    insert_sql = sql.SQL("""
        INSERT INTO {table}
        (filename, centroid, point_cloud_100_x, point_cloud_100_y, point_cloud_100_z, 
         point_cloud_5000_x, point_cloud_5000_y, point_cloud_5000_z)
        VALUES (%s, ST_MakePoint(%s, %s, %s), %s, %s, %s, %s, %s, %s)
    """).format(table=sql.Identifier(table))

    cur.execute(insert_sql, (data['filename'], data['centroid'][0], data['centroid'][1], data['centroid'][2],
                             data['point_cloud_100_x'], data['point_cloud_100_y'], data['point_cloud_100_z'],
                             data['point_cloud_5000_x'], data['point_cloud_5000_y'], data['point_cloud_5000_z']))

    conn.commit()

    print(f"Processed {data['filename']}. Point Cloud Size: {len(data['point_cloud_5000_x'])}")

    cur.close()


def process_and_upload_files_from_azure_blob(container_name):
    blob_container_client = blob_service_client.get_container_client(container_name)
    blobs = blob_container_client.list_blobs()

    blobs_list = list(blobs)
    total_blobs = len(blobs_list)
    file_count = 0

    print(f"Starting processing of {total_blobs} blobs...")

    conn = connect_to_db()

    # Ensure the table exists before processing blobs
    create_table_if_not_exists(conn, table)

    for blob_chunk in chunks(blobs_list):
        batch_data = []
    
        for blob in blob_chunk:
            data = process_blob(blob)
            batch_data.append(data)
        
            file_count += 1
            print(f"Processed file {file_count}/{total_blobs}: {data['filename']}")

        for item in batch_data:
            upload_point_cloud_to_db(conn, table, item)

    conn.close()

    print(f"Total number of files processed: {file_count}/{total_blobs}")

def main():
    # Establish a database connection
    conn = connect_to_db()
    
    # Ensure the table exists
    create_table_if_not_exists(conn, table)
    
    # Close the initial connection
    conn.close()

    # Process the blobs and upload to the database
    process_and_upload_files_from_azure_blob(container_name)

if __name__ == "__main__":
    table = "11iguscad"
    container_name = 'bearings'
    main()