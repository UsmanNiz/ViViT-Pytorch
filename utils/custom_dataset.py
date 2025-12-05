import decord
from decord import cpu, gpu
import numpy as np
import random

# import torchvision
import os
import torch
import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as F



class DecordInit(object):
	"""Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

	def __init__(self, num_threads=1, **kwargs):
		# Reduced to 1 thread to avoid threading issues with get_batch
		# Error -11 (EAGAIN) occurs when multiple threads try to decode simultaneously
		self.num_threads = num_threads
		self.ctx = decord.cpu(0)
		#decord.bridge.set_bridge('torch')
		self.kwargs = kwargs
		
	def __call__(self, filename):
		"""Perform the Decord initialization.
		Args:
			results (dict): The resulting dict to be modified and passed
				to the next transform in pipeline.
		"""
		# print(f"[DEBUG DecordInit] Attempting to open video: {filename}")
		if not os.path.exists(filename):
			# print(f"[DEBUG DecordInit] ERROR: File does not exist: {filename}")
			raise FileNotFoundError(f"Video file not found: {filename}")
		
		# Check file size
		file_size = os.path.getsize(filename)
		# print(f"[DEBUG DecordInit] File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
		
		if file_size == 0:
			# print(f"[DEBUG DecordInit] ERROR: File is empty: {filename}")
			raise ValueError(f"Video file is empty: {filename}")
		
		try:
			# print(f"[DEBUG DecordInit] Creating VideoReader with num_threads={self.num_threads}")
			reader = decord.VideoReader(filename,
										ctx=self.ctx,
										num_threads=self.num_threads)
			frame_count = len(reader)
			# print(f"[DEBUG DecordInit] Successfully opened video: {filename}, frames: {frame_count}")
			return reader
		except Exception as e:
			print(f"[DEBUG DecordInit] ERROR creating VideoReader for {filename}: {type(e).__name__}: {str(e)}")
			raise

	def __repr__(self):
		repr_str = (f'{self.__class__.__name__}('
					f'num_threads={self.num_threads})')
		return repr_str

class CustomDataset(torch.utils.data.Dataset):
	"""Load the video files
	
	Args:
		data_path (string): Path to dataset.
  		label_path (string): Path to label
		num_frames: Number of input frame to be extracted per video
		transform: data augmentation
		sample_method: tubelet or uniform sampling
  		blackbar_check: check for existance of black bar in input video
	"""

	def __init__(self,
				 data_path,
				 label_path=None,
				 num_frames=64,
				 tubelet_size=2,
				 transform=None,
				 sample_method="tubelet",
     			 blackbar_check=None):
		# self.configs = configs
		self.labels = pd.read_csv(label_path) if label_path != None else None
		self.label_path = label_path
		self.data = sorted(os.listdir(data_path))
		self.data_path = data_path
		self.tubelet_size = tubelet_size
		self.transform = transform
		self.sample_method = sample_method
		self.num_frames = num_frames
		self.v_decoder = DecordInit()
		self.blackbar_check = blackbar_check

	def __getitem__(self, index):
		while True:
			try:
				vid = self.data[index]
				path = os.path.join(self.data_path, vid)
				blackbak_crop = None
				if self.blackbar_check != None:
					# print(f"[DEBUG __getitem__] Running blackbar_check...")
					blackbak_crop = self.blackbar_check(path)
				
				# print(f"[DEBUG __getitem__] Creating video reader...")
				v_reader = self.v_decoder(path)
				total_frames = len(v_reader)
				sample_length = self.tubelet_size * self.num_frames
				# Sampling video frames
				if self.sample_method == "tubelet":
					rand_end = max(0, total_frames - sample_length - 1)
					begin_index = random.randint(0, rand_end)
					end_index = min(begin_index + sample_length, total_frames)
					assert end_index-begin_index >= sample_length
					frame_indice = np.linspace(begin_index, end_index-1, self.num_frames, dtype=int)
				elif self.sample_method == "uniform_sampling":
					frame_indice = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
				video = v_reader.get_batch(frame_indice).asnumpy()
				del v_reader
				# print(f"[DEBUG __getitem__] Successfully loaded video: {vid}")
				break
				
			except Exception as e:
				print(e)
				index = random.randint(0, len(self.data) - 1)
	
		# Video align transform: T C H W
		# print(f"[DEBUG __getitem__] Applying transforms to video: {vid}")
		with torch.no_grad():
			if self.label_path != None:
				label = self.labels[self.labels["fname"] == vid]["class_id"].item()
				video = torch.tensor(video, dtype=torch.float32)
			else:
				label = None
				video = torch.tensor(video)
			video = video.permute(0,3,1,2)
			video = torch.div(video, 255)
			
			if blackbak_crop is not None:
				# print(f"[DEBUG __getitem__] Applying blackbar crop: {blackbak_crop}")
				video = F.crop(video,blackbak_crop[3],blackbak_crop[2],blackbak_crop[1],blackbak_crop[0])
				
			if self.transform is not None:
				# print(f"[DEBUG __getitem__] Applying transform...")
				# print(f"[DEBUG __getitem__] Video tensor shape BEFORE transform: {video.shape}, dtype: {video.dtype}")
				video = self.transform(video)

		data_out = (video,label) if label != None else video
		return  data_out
		

	def __len__(self):
		return len(self.data)

	def set_transform(self,transform):
		self.transform = transform
	
	def collate_fn(self, batch):
		return tuple(zip(*batch))






