
def save_pt_as_fcsv(infile, outfile):

    pts = eval(open(infile).read())
    content = '# Markups fiducial file version = 5.2\n' \
        '# CoordinateSystem = LPS\n' \
        '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n'
    
    content += '\n'.join([
        f"{_id},{x},{y},{z},0,0,0,1,1,1,1,F-{label.replace('Marker ', '')},,,," for _id, (_num, label, (x,y,z)) in enumerate(pts, start=1)
    ])

    with open(outfile, 'w') as f:
        f.write(content)

    print(f'Saved to {outfile}')

if __name__ == "__main__":
    from pathlib import Path
    
    data_dir = Path('data')

    for f in data_dir.iterdir():
        if not f.name.startswith('PMCC'):
            continue
        
        save_pt_as_fcsv(f/'p1.txt', f/'p1.fcsv')
        save_pt_as_fcsv(f/'p2.txt', f/'p2.fcsv')
