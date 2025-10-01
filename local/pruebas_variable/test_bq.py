import os
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "steel-autonomy-472502-f3")

client = bigquery.Client(project=PROJECT_ID)

datasets = list(client.list_datasets())
if datasets:
    print("Datasets en el proyecto:")
    for ds in datasets:
        print(f" - {ds.dataset_id}")
else:
    print("No se encontraron datasets en el proyecto.")