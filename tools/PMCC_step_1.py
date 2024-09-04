# Sort the DCM files into two scans, each with a CT and RTStruct

from pmcc_rmse import sort_files, transfer_files
from pathlib import Path

data_dir = Path(r'\\petermac.org.au\shared\ImageStore\Reirradiation\Lung')
save_dir = Path(r'P:\yusun\self-supervised-anatomical-embedding-v2\data\lung')
for folder in data_dir.iterdir():
    # Skip 300xxx folders
    if folder.name.startswith('300'): continue
    
    each_save_dir = save_dir / folder.name
    if not each_save_dir.exists(): each_save_dir.mkdir(exist_ok=True)

    file_info = sort_files(folder)
    transfer_files(file_info, each_save_dir)
    