import os
import urllib.request
import shutil
import zipfile

def download_and_extract_opensmile(opensmile_version):
    opensmile_folder = f"opensmile-{opensmile_version}"
    opensmile_zip_file = f"{opensmile_folder}.zip"
    opensmile_url = f"https://www.audeering.com/download/{opensmile_zip_file}?wpdmdl={opensmile_version}"

    if not os.path.exists(opensmile_folder):
        # Download openSMILE zip file
        print(f"Downloading openSMILE version {opensmile_version}...")
        with urllib.request.urlopen(opensmile_url) as response, open(opensmile_zip_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"Downloaded {opensmile_zip_file}")

        # Extract zip file
        print(f"Extracting {opensmile_zip_file}...")
        with zipfile.ZipFile(opensmile_zip_file, 'r') as zip_ref:
            zip_ref.extractall()
        print(f"Extracted to {opensmile_folder}")

        # Clean up zip file
        os.remove(opensmile_zip_file)
        print(f"Removed {opensmile_zip_file}")
    else:
        print(f"openSMILE version {opensmile_version} is already downloaded.")

# Use the function
download_and_extract_opensmile('2.3.0')
