from gaps_dataset import gaps
from pathlib import Path

login_v1 = 'gapsroG8(^#g6'
login_v2 = 'gapsro2s;i2A*7'
login_10m = 'gapsro3Z=1Yb%7'
output_dir = '/home/vivek/repos/GenFaD/data/GAPS_v2'

gaps.download(login=login_v2,
                output_dir=str(output_dir),
                version=2,
                patchsize='images',
                issue='NORMvsDISTRESS_50k',
                debug_outputs=True)