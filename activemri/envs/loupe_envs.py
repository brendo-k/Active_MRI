import pathlib
import torch
from torch.utils.data import DataLoader, random_split

import activemri.data.singlecoil_knee_data as scknee_data
from activemri.data.real_knee_data import RealKneeData
from activemri.data.real_brain_data import RealBrainData
from activemri.envs.envs import ActiveMRIEnv


