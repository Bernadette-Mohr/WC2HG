from tqdm.auto import tqdm
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True
from openpathsampling.experimental.storage import Storage


old_storage = Storage(f'/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC/DNAWC_DNAWC_100fs.db', 'r')
new_storage = Storage(f'/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC/DNAWC_100fs_decorrelated.db', 'w')

for cv in old_storage.storable_functions:
    cv.preload_cache()

for obj in tqdm(old_storage.simulation_objects):
    new_storage.save(obj)

try:
    # for idx, step in enumerate(tqdm(old_storage.steps)):
    for idx in tqdm([0, 26, 34, 63, 72, 80, 91, 97, 106, 108, 131, 149, 162, 168, 182]):
        new_storage.save(old_storage.steps[idx])
except Exception as e:
    print(f'Unable to load step {idx} from storage: {e.__class__}: {str(e)}')
    pass

new_storage.sync_all()
new_storage.close()

# scheme = new_storage.schemes[0]
# scheme.move_summary(steps=new_storage.steps)  # just returns some statistics
