from dataclasses import dataclass
from ..constants import *

@dataclass
class SERDataLoaderArtifacts:
    emotion_map = EMOTION_MAP
    ser_path_extension = SER_PATH_EXENSION
    ser_res_type = SER_RES_TYPE
    