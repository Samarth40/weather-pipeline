import os
import io

class LocalBlobInfo:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<LocalBlobInfo: {self.name}>"


class LocalBlobClient:
    def __init__(self, filepath):
        self.filepath = filepath

    def download_blob(self):
        parent = self

        class DownloadStream:
            def readall(self):
                with open(parent.filepath, "rb") as f:
                    return f.read()

            def readinto(self, buffer):
                data = self.readall()
                if hasattr(buffer, "write"):
                    buffer.write(data)
                elif hasattr(buffer, "__setitem__"):
                    buffer[:len(data)] = data
                return len(data)

        return DownloadStream()

    def upload_blob(self, data, overwrite=True):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "wb") as f:
            if hasattr(data, "getvalue"):
                val = data.getvalue()
                if isinstance(val, str):
                    val = val.encode("utf-8")
                f.write(val)
            elif hasattr(data, "read"):
                if hasattr(data, "seek"):
                    data.seek(0)
                val = data.read()
                if isinstance(val, str):
                    val = val.encode("utf-8")
                f.write(val)
            elif isinstance(data, str):
                f.write(data.encode("utf-8"))
            elif isinstance(data, bytes):
                f.write(data)
            else:
                f.write(bytes(data))


class LocalContainerClient:
    def __init__(self, container_path):
        self.container_path = container_path
        os.makedirs(self.container_path, exist_ok=True)

    def list_blobs(self):
        if not os.path.exists(self.container_path):
            return []
        items = []
        for fname in os.listdir(self.container_path):
            full_path = os.path.join(self.container_path, fname)
            if os.path.isfile(full_path):
                items.append(LocalBlobInfo(fname))
        return items

    def get_blob_client(self, blob_name):
        return LocalBlobClient(os.path.join(self.container_path, blob_name))


class LocalBlobServiceClient:
    def __init__(self, base_dir=None):
        if not base_dir:
            # Anchor to MLOps_Project/local_data
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, "local_data")
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def get_container_client(self, container_name):
        return LocalContainerClient(os.path.join(self.base_dir, container_name))

    def get_blob_client(self, container_name, blob_name):
        return LocalBlobClient(os.path.join(self.base_dir, container_name, blob_name))


def get_blob_service_client(connection_string=None):
    """
    Returns an Azure BlobServiceClient if a valid Azure connection string is present,
    otherwise returns a LocalBlobServiceClient that persists files locally.
    """
    if not connection_string:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    # If connection string is empty, 'local', or not pointing to Azure, use local emulator
    if not connection_string or connection_string.strip().lower() in ("local", "none", "mock", "") or not connection_string.startswith("DefaultEndpointsProtocol"):
        print("[Local Storage] Using offline local filesystem storage adapter at ./local_data")
        return LocalBlobServiceClient()

    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient.from_connection_string(connection_string)
