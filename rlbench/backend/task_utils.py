import os
import numpy as np
from pyrep.objects.shape import Shape

keep = [0, 1, 10, 23, 24, 32, 35, 40, 55, 100]

def sample_procedural_objects(task_base, num_samples, mass=0.1):
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '../assets/procedural_objects')
    samples = [str(k).zfill(3) for k in keep]
    samples = np.random.choice(samples, num_samples, replace=True)
    # samples = np.random.choice(
    #     os.listdir(assets_dir), num_samples, replace=True)
    created = []
    for s in samples:
        respondable = os.path.join(assets_dir, s, s + '_coll.obj')
        visual = os.path.join(assets_dir, s, s + '.obj')
        resp = Shape.import_mesh(respondable, scaling_factor=0.0075)
        vis = Shape.import_mesh(visual, scaling_factor=0.0075)
        resp.set_renderable(False)
        vis.set_renderable(True)
        vis.set_parent(resp)
        vis.set_dynamic(False)
        vis.set_respondable(False)
        resp.set_dynamic(True)
        resp.set_mass(mass)
        resp.set_respondable(True)
        resp.set_model(True)
        resp.set_parent(task_base)
        created.append(resp)
    return created
