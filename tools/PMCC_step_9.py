# Generate .bat for transformix
# Warp pt1 to pt2

from pathlib import Path

template = 'c:\elastix-5.0.0-win64\transformix.exe -tp ..\registration\01\TransformParameters.0.txt -def PMCC_ReIrrad_01\pt1_elastix_input.txt -out PMCC_ReIrrad_01'

point_dir = Path(r'P:\yusun\target_registration_error\points')
reg_dir = Path(r'P:\yusun\target_registration_error\regisration_ct1_fixed')

bin_path = r'c:\elastix-5.0.0-win64\transformix.exe'

cmds = []
for f in point_dir.iterdir():
    each_reg_dir = reg_dir / f.stem[-2:]

    each_save_dir = f / 'warped_point'
    each_save_dir.mkdir(exist_ok=True)
    
    tfm_file = each_reg_dir / 'TransformParameters.0.txt'
    pt_file = f / 'pt1_elastix_input.txt'
    cmd_tfx = f'{bin_path} -tp {tfm_file} -def {pt_file} -out {each_save_dir}\n\n'
    print(cmd_tfx)
    cmds.append(cmd_tfx)

with open(point_dir/'warp_points.bat', 'w') as f:
    f.writelines(cmds)