import os
import gcsfs
import google.auth

CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

def authenticate_gcp(project_id: str | None = None, bucket_test: str | None = "isign_bucket"):
    """
    Deterministic GCS auth using Application Default Credentials (ADC).
    Avoids gcsfs falling back to anonymous in some notebook/import contexts.
    """
    adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    if not os.path.exists(adc_path):
        raise RuntimeError(
            f"ADC not found at {adc_path}\n"
            f"Run: gcloud auth application-default login --scopes={CLOUD_SCOPE}"
        )

    creds, proj = google.auth.default(scopes=[CLOUD_SCOPE])
    fs = gcsfs.GCSFileSystem(project=project_id or proj, token=creds)

    if bucket_test:
        # force a real call so any auth issue shows up here
        fs.ls(bucket_test)

    print("GCS filesystem initialized (explicit ADC).")
    return fs