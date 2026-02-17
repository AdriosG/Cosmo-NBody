import numpy as np
import cupy as cp
from tqdm import tqdm 
from src.utils.visualizer import get_mvp_matrix, colorize_frame
from src.kernels.render_kernels import render_density
from PIL import Image
import subprocess

class Camera:
    def __init__(self, n_steps, resolution, config):

        """
        Class to handle all the rendering part of the simulation.
        - capture(args) to generate a frame
        - get_frames(args) to send the frames from VRAM to DISK
        - make_video(args) simple execution of ffmpeg to make video from frames.
         
        :param n_steps: int, number of simulation steps (frames)
        :param resolution: int, resolution of the render (res x res)
        :param config: Object storing constants
        """

        self.config = config
        self.n = n_steps
        self.video_buffer = np.zeros((n_steps, resolution, resolution, 3), dtype=np.uint8)
        self.grid_d = cp.zeros((resolution, resolution), dtype=cp.float32)
        self.box_center = (self.config.BOX_SIZE/2, self.config.BOX_SIZE/2, self.config.BOX_SIZE/2)

    def capture(self, step, px_d, py_d, pz_d):

        """
        Capture a frame
        
        Param:

        :param step: int, current step
        :px_d, py_d, pz_d : DeviceArray, stores bodies positions.
        """

        self.grid_d.fill(0)

        mvp = get_mvp_matrix(step, self.box_center, self.config.BOX_SIZE, self.config)

        render_density[self.config.BPG, self.config.TPB](px_d, py_d, pz_d, self.grid_d, mvp, self.config.RES)

        frame = colorize_frame(self.grid_d)

        return frame

    def get_frames(self):
        """
        Sends back the frame from the GPU the CPU and write them to the disk.
        Could be optimised if implemented with cncurrence.
        """

        for i in tqdm(range(self.n)):
            Image.fromarray(self.video_buffer[i]).save(f"{self.config.OUTPUT_DIR}/frame_{i:04d}.png")


    def make_video(self):

        """
        Generates video using ffmpeg
        """

        subprocess.run(['ffmpeg', '-y',
                        '-loglevel', 'warning', 
                        '-framerate', '60', 
                        '-i', f'{self.config.OUTPUT_DIR}/frame_%04d.png', 
                        '-c:v', 'libx264', 
                        '-pix_fmt', 'yuv420p', 'final_cosmic_v3.mp4'])