from tqdm.auto import tqdm
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True
from openpathsampling.experimental.storage import Storage


# old_storage = Storage(f'/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC2MAT/DNAWC2MAT_n_steps_50.db', 'r')
# new_storage = Storage(f'/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC2MAT/DNAWC2MAT_n_steps_50_decorrelated.db', 'w')

# for cv in tqdm(old_storage.storable_functions, desc='Preloading cache'):
#     cv.preload_cache()

# for obj in tqdm(old_storage.simulation_objects, desc='Copying simulation objects'):
#     new_storage.save(obj)


# try:
#     # for idx, step in enumerate(tqdm(old_storage.steps)):
#     # for idx in tqdm([0, 26, 34, 63, 72, 80, 91, 97, 106, 108, 131, 149, 162, 168, 182]):  # DNAWC
#     for idx in tqdm([0, 22, 33, 47, 54, 104, 111, 134, 156, 172, 198], desc='Decorrelated paths'):  # DNAWC2MAT
#         new_storage.save(old_storage.steps[idx])
# except Exception as e:
#     print(f'Unable to load step {idx} from storage: {e.__class__}: {str(e)}')
#     pass

# new_storage.sync_all()
# new_storage.close()

# scheme = new_storage.schemes[0]
# scheme.move_summary(steps=new_storage.steps)  # just returns some statistics

storage = Storage(f'/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC2MAT/DNAWC2MAT_n_steps_50_decorrelated.db', 'r')
print(storage.steps[2].active[0].trajectory)
cvs = dict()
for cv in storage.storable_functions:
    cvs[cv.name] = cv
print(cvs)

for obj in storage.simulation_objects:
    print(obj.name)
    # if obj.name == '[MDTrajTopology]':
    #     print(obj)
    # if obj.name == '[OneWayShootingMoveScheme]':
    #     print(obj)
    # elif obj.name == 'OMM_engine':
    #    print(obj)

print(storage.networks[0])
print(storage.engines[2])
