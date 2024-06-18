

import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print(sys.path)
from pypef.dca.gremlin_inference import GREMLIN


single_point_mut_data = os.path.abspath(os.path.join(os.path.dirname(__file__), f"single_point_dms_mut_data.json"))
higher_mut_data = os.path.abspath(os.path.join(os.path.dirname(__file__), f"higher_point_dms_mut_data.json"))


with open(single_point_mut_data, 'r') as fh:
    mut_data = json.loads(fh.read())



for i, (dset_key, dset_paths) in enumerate(mut_data.items()):
    print(i+1, dset_key, dset_paths['CSV_path'], dset_paths['MSA_path'], dset_paths['MSA_path'])

