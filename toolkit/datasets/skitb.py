from __future__ import absolute_import, print_function

import six
import os
import numpy as np

class SkiTB(object):
    """The SkiTB <http://machinelearning.uniud.it/datasets/skitb/> Benchmark.

    Publication:
        ``Tracking Skiers from the Top to the Bottom``,
        Matteo Dunnhofer, Luca Sordi, Niki Martinel and Christian Micheloni, WACV 2024.

    """

    def __init__(self, root_dir, discipline='AL', mode='MC', test_videos=None):
        super(SkiTB, self).__init__()

        self.root_dir = root_dir
        self.discipline_root_dir = os.path.join(self.root_dir, discipline)

        self.mode = mode
        if test_videos is None:
            self.seq_names = sorted(next(os.walk(self.discipline_root_dir))[1], key=lambda x: int(x[2:]))
        else:
            self.seq_names = test_videos

        self.seq_dirs = [os.path.join(self.discipline_root_dir, n) for n in self.seq_names]
        
        if self.mode == 'SC':
            seq_names_st = []
            for seq_name in self.seq_names:
                if len(seq_name.split('_')) == 1:
                    idxs = sorted(os.listdir(os.path.join(self.discipline_root_dir, seq_name, 'ST')), key=lambda x: int(x))
    
                    for idx in idxs:
                        seq_names_st.append(f'{seq_name}_{idx}')
                else:
                    seq_names_st.append(seq_name)

            self.seq_names = seq_names_st
            self.seq_dirs = [os.path.join(self.discipline_root_dir, n.split('_')[0]) for n in self.seq_names]


    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        if self.mode == 'MC':
            frame_idxs = np.loadtxt(os.path.join(self.seq_dirs[index], 'MC', 'frames.txt'), delimiter='\n')
            anno = np.loadtxt(os.path.join(self.seq_dirs[index], 'MC', 'boxes.txt'), delimiter=',')
            visibilities = np.loadtxt(os.path.join(self.seq_dirs[index], 'MC', 'visibilities.txt'), delimiter='\n')
            camera_idxs = np.loadtxt(os.path.join(self.seq_dirs[index], 'MC', 'cameras.txt'), delimiter='\n').astype(np.int64)
            img_files = [os.path.join(self.seq_dirs[index], 'frames', f'{int(fi):05d}.jpg') for fi in frame_idxs]
        else:
            seq_idx = self.seq_names[index].split('_')[-1]
            frame_idxs = np.loadtxt(os.path.join(self.seq_dirs[index], 'SC', seq_idx, 'frames.txt'), delimiter='\n', ndmin=1)
            anno = np.loadtxt(os.path.join(self.seq_dirs[index], 'SC', seq_idx, 'boxes.txt'), delimiter=',', ndmin=2)
            visibilities = np.loadtxt(os.path.join(self.seq_dirs[index], 'SC', seq_idx, 'visibilities.txt'), delimiter='\n', ndmin=1)
            camera_idxs = np.zeros(anno.shape[0]) + int(seq_idx)
            img_files = [os.path.join(self.seq_dirs[index], 'frames', f'{int(fi):05d}.jpg') for fi in frame_idxs]
        
        assert len(img_files) == len(anno) == len(visibilities) == len(camera_idxs), (len(img_files), len(anno), len(visibilities), len(camera_idxs))
        assert anno.shape[1] == 4

        return img_files, anno, visibilities, camera_idxs

    def __len__(self):
        return len(self.seq_names)
        
    

