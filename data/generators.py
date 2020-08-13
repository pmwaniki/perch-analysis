from itertools import cycle
import numpy as np



def multitask_generator(perch_path,perch_labs,chestray_path,chestray_labs):
    chestray_index = np.array(range(len(chestray_path)))
    np.random.shuffle(chestray_index)
    perch_index = np.array(range(len(perch_path)))
    np.random.shuffle(perch_index)
    zipped_index=zip(chestray_index,cycle(perch_index))
    for chestray_i,perch_i in zipped_index:
        yield perch_path[perch_i],perch_labs[perch_i],chestray_path[chestray_i],chestray_labs[chestray_i]

def multitask_ensemble_generator(perch_path,perch_labs,perch_rev,chestray_path,chestray_labs):
    chestray_index = np.array(range(len(chestray_path)))
    np.random.shuffle(chestray_index)
    perch_index = np.array(range(len(perch_path)))
    np.random.shuffle(perch_index)
    zipped_index=zip(chestray_index,cycle(perch_index))
    for chestray_i,perch_i in zipped_index:
        yield perch_path[perch_i],perch_labs[perch_i],perch_rev[perch_i],chestray_path[chestray_i],chestray_labs[chestray_i]