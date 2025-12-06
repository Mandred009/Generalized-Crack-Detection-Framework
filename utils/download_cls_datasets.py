from pathlib import Path
import pandas as pd
from argparse import ArgumentParser
import sys
import re

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Error: kaggle package not found. Install it with: pip install kaggle")
    sys.exit(1)
    
from utils.misc_utils import bcolors
    
def setup_kaggle_api()->KaggleApi:
    """
    Initialize and authenticate Kaggle API.
    
    Returns:
        KaggleApi: Authenticated API instance
    """
    api = KaggleApi()
    
    try:
        api.authenticate()
        print(f"{bcolors.OKGREEN}Kaggle API authentication successful{bcolors.ENDC}")
        return api
    except Exception as e:
        print(f"Error: Kaggle API authentication failed: {e}")
        print("\nPlease ensure:")
        print("1. You have kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
        print("2. The file has correct permissions (chmod 600 ~/.kaggle/kaggle.json on Linux/Mac)")
        print("3. Your kaggle.json contains valid API credentials")
        sys.exit(1)
        
def extract_dataset_info(kaggle_link: str):
    """
    Extract username and dataset name from Kaggle URL or direct reference.
    
    Args:
        kaggle_link (str): Kaggle dataset URL or username/dataset-name format
        
    Returns:
        tuple: (username, dataset_name)
    """
    # Handle direct format like "username/dataset-name"
    if "/" in kaggle_link and not kaggle_link.startswith("http"):
        parts = kaggle_link.strip().split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
    
    # Handle full URL format
    patterns = [
        r'kaggle\.com/datasets/([^/]+)/([^/?]+)',
        r'kaggle\.com/([^/]+)/([^/?]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, kaggle_link)
        if match:
            return match.group(1), match.group(2)
    
    raise ValueError(f"Could not parse dataset info from: {kaggle_link}")

def download_dataset(
    api: KaggleApi, 
    username: str, 
    dataset_name: str, 
    download_path: Path, 
    unzip: bool=True
):
    """
    Download and optionally extract Kaggle dataset.
    
    Args:
        api (KaggleApi): Authenticated Kaggle API instance
        username (str): Dataset owner username
        dataset_name (str): Dataset name
        download_path (Path): Local directory to download to
        unzip (bool): Whether to extract zip files
    """
    dataset_ref = f"{username}/{dataset_name}"
    
    print(f"Downloading dataset: {dataset_ref}")
    print(f"Download directory: {download_path.absolute()}")
    
    try:
        # Download dataset
        api.dataset_download_files(
            dataset_ref,
            path=str(download_path),
            unzip=unzip
        )
        
        print(f"{bcolors.OKGREEN}Dataset downloaded successfully to: {download_path.absolute()}{bcolors.ENDC}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Try to get more specific error information
        try:
            dataset_info = api.dataset_metadata(dataset_ref, str(download_path))
            print(f"Dataset exists: {dataset_info.title}")
            print("This might be a permissions or network issue.")
        except:
            print(f"Dataset '{dataset_ref}' may not exist or may be private.")
        
        return

def process_kaggle_dataset(
    dataset_name: str, 
    url: str, 
    output_dir: Path, 
    no_unzip: bool
):
    '''
    Download the dataset in the output directory under the `dataset_name` subdir.
    '''
    dataset_output_dir = output_dir / dataset_name.replace('/', '_').replace(' ', '-')
    dataset_output_dir.mkdir(exist_ok=True, parents=True)
    
    username, dataset_name = extract_dataset_info(url)
    print(f"Dataset: {username}/{dataset_name}")
    
    # Setup Kaggle API
    api = setup_kaggle_api()
    
    # Download dataset
    download_dataset(
        api = api,
        username = username,
        dataset_name = dataset_name,
        download_path = dataset_output_dir,
        unzip = not no_unzip
    )
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset-csv', type=Path, required=True, help='path to dataset csv')
    parser.add_argument('--output-dir', type=Path, required=True, help='path to output directory')
    parser.add_argument('--no-unzip', action='store_true', help='Set flag to NOT unzip the datasets')
    args = parser.parse_args()
    
    output_dir: Path = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    dataset_list = pd.read_csv(args.dataset_csv).to_dict()
        
    for row_idx in dataset_list['name']:
        name = dataset_list['name'][row_idx]
        url = dataset_list['link'][row_idx]
        
        if 'www.kaggle.com' in url:
            process_kaggle_dataset(name, url, output_dir, args.no_unzip)
    
if __name__ == '__main__':
    main()