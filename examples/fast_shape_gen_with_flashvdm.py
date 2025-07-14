# HY3DGEN_DEBUG=1 USE_SAGEATTN=1 python3 examples/fast_shape_gen_with_flashvdm.py
# HY3DGEN_DEBUG=1 USE_SAGEATTN=0 python3 examples/fast_shape_gen_with_flashvdm.py

import os
import time

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
    subfolder='hunyuan3d-dit-v2-0-turbo',
    use_safetensors=True,
)
pipeline.enable_flashvdm()
# pipeline.compile()

image_path = 'assets/demo.png'
image = Image.open(image_path).convert("RGBA")
if image.mode == 'RGB':
    rembg = BackgroundRemover()
    image = rembg(image)


def run():
    return pipeline(
        image=image,
        num_inference_steps=5,
        octree_resolution=380,
        num_chunks=200000,
        generator=torch.manual_seed(12345),
        output_type='trimesh'
    )[0]


save_dir = 'tmp/results/'
os.makedirs(save_dir, exist_ok=True)

for it in range(2):
    start_time = time.time()
    mesh = run()
    print("--- %s seconds ---" % (time.time() - start_time))
    mesh.export(f'{save_dir}/run_{it}.glb')
