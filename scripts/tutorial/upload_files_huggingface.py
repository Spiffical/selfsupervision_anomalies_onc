import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/home/sbialek/ONC/selfsupervision_anomalies_onc/data/datasets",
    repo_id="merileo/onc-ssl-tutorial",
    repo_type="model",
)
