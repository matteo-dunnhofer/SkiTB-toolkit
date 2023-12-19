from __future__ import absolute_import, division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import itertools
import json
from PIL import Image
import time
import copy
import glob

from ..datasets import SkiTB
from ..utils.metrics import rect_iou, center_error, normalized_center_error
from ..utils.viz import show_frame

class ExperimentSkiTB(object):
    r"""Experiment pipeline and evaluation toolkit for the SkiTB dataset.

    Args:
        root_dir (string): Root directory of OTB dataset.
        version (integer or string): Specify the benchmark version, specify as one of
            ``2013``, ``2015``, ``tb50`` and ``tb100``. Default is ``2015``.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir, discipline='AL', mode='LT', test_videos=None, result_dir='./results', report_dir='./reports'):
        super(ExperimentSkiTB, self).__init__()
        self.root_dir = root_dir
        self.discipline = discipline
        self.mode = mode
        self.test_videos = test_videos

        self.dataset = SkiTB(root_dir, discipline=self.discipline, mode=self.mode, test_videos=self.test_videos)
        self.result_dir = os.path.join(result_dir, 'SkiTB')
        self.report_dir = os.path.join(report_dir, 'SkiTB')
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.nbins_nce = 51
        self.nbins_gsr = 51

        self.conf_thresholds = np.arange(0, 100, dtype=np.float32) / 100


    def run(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (
            tracker.name, type(self.dataset).__name__))

        # loop over the complete dataset
        for s, (img_files, anno, visibilities, camera_idxs) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_dir = os.path.join(self.result_dir, tracker.name, self.mode, self.discipline, 'ope', seq_name)
            if os.path.exists(os.path.join(record_dir, 'boxes.txt')):
                print('  Found results, skipping', seq_name)
                continue

            # tracking loop
            boxes, confidences, times = tracker.track(
                img_files, anno[0, :], visualize=visualize)
            assert len(boxes) == len(confidences) == len(anno)

            # record results
            self._record(record_dir, boxes, confidences, times)


    def report(self, tracker_names, fn_note=None):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        if self.test_videos is None:
            report_file = os.path.join(report_dir, f'performance-{self.mode}-{self.discipline}-ope.json')
        else:
            report_file = os.path.join(report_dir, f'performance-{self.mode}-{self.discipline}-{fn_note}-ope.json')
        
        performance = {}
        for name in tracker_names:
            print('Evaluating', name)
            seq_num = len(self.dataset)
            
            f_scores = np.zeros((seq_num, len(self.conf_thresholds)))
            precisions = np.zeros((seq_num, len(self.conf_thresholds)))
            recalls = np.zeros((seq_num, len(self.conf_thresholds)))
            gen_succ_rob_curve = np.zeros((seq_num, self.nbins_gsr))
            speeds = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, (_, anno, visibilities, _) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                
                record_dir = os.path.join(self.result_dir, name, self.mode, self.discipline, 'ope', seq_name)

                boxes = np.loadtxt(os.path.join(record_dir, 'boxes.txt'), delimiter=',')
                confidences = np.loadtxt(os.path.join(record_dir, 'confidences.txt'), delimiter=',')
                times = np.loadtxt(os.path.join(record_dir, 'times.txt'))

                boxes[0] = anno[0, :]
                
                assert len(boxes) == len(confidences) == len(anno)

                ious, _ = self._calc_metrics(boxes[visibilities > 0], anno[visibilities > 0])
                gen_succ_rob_curve[s] = self._calc_curves_robustness(ious)

                # vot-lt metrics
                for ct, conf_tresh in enumerate(self.conf_thresholds):
                    ious, _ = self._calc_metrics(boxes[confidences >= conf_tresh], anno[confidences >= conf_tresh])
                    precisions[s,ct] = np.sum(ious) / np.sum(confidences >= conf_tresh)
                    recalls[s,ct] = np.sum(ious) / np.sum(visibilities > 0)
                    f_scores[s,ct] = (2.0 * precisions[s,ct] * recalls[s,ct]) / (precisions[s,ct] + recalls[s,ct])
              
                # calculate average tracking speed
                times = times[times > 0]
                if len(times) > 0:
                    speeds[s] = np.mean(1. / times)

                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'f_score' : float(np.max(f_scores[s])),
                    'precision' : precisions[s, np.argmax(f_scores[s])],
                    'recall' : recalls[s, np.argmax(f_scores[s])],
                    'f_score_thresh' : float(self.conf_thresholds[np.argmax(f_scores[s])]),
                    'generalized_success_robustness_curve': gen_succ_rob_curve[s].tolist(),
                    'generalized_success_robustness_score': np.mean(gen_succ_rob_curve[s]),
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            f_score_curve = np.mean(f_scores, axis=0)
            precision_curve = np.mean(precisions, axis=0)
            recall_curve = np.mean(recalls, axis=0) 
            gen_succ_rob_curve = np.mean(gen_succ_rob_curve, axis=0)
            gen_succ_rob_score = np.mean(gen_succ_rob_curve)

            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'f_score' : float(np.max(f_score_curve)),
                'precision' : precision_curve[np.argmax(f_score_curve)],
                'recall' : recall_curve[np.argmax(f_score_curve)],
                'f_score_thresh' : float(self.conf_thresholds[np.argmax(f_score_curve)]),
                'generalized_success_robustness_curve': gen_succ_rob_curve.tolist(),
                'generalized_success_robustness_score': gen_succ_rob_score,
                'speed_fps': avg_speed})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        
        return performance

    
    def _record(self, record_dir, boxes, confidences, times):
        
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)

        record_file_boxes = os.path.join(record_dir, 'boxes.txt')
        np.savetxt(record_file_boxes, boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file_boxes):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file_boxes, boxes, fmt='%.3f', delimiter=',')
        print('  Box results recorded at', record_file_boxes)

        record_file_confs = os.path.join(record_dir, 'confidences.txt')
        np.savetxt(record_file_confs, confidences, fmt='%.8f', delimiter='\n')
        while not os.path.exists(record_file_confs):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file_confs, confidences, fmt='%.8f', delimiter='\n')
        print('  Conf results recorded at', record_file_confs)

        # record running times
        record_file_times = os.path.join(record_dir, 'times.txt')
        np.savetxt(record_file_times, times, fmt='%.8f', delimiter='\n')
        while not os.path.exists(record_file_times):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file_times, times, fmt='%.8f', delimiter='\n')
        print('  Time results recorded at', record_file_times)


    def _calc_metrics(self, boxes, anno):
        ious = []
        norm_center_errors = [] 

        for box, a in zip(boxes, anno):
            if a[0] < 0 and a[1] < 0 and a[2] < 0 and a[3] < 0:
                continue
            else:
                ious.append(rect_iou(np.array([box]), np.array([a]))[0])
                norm_center_errors.append(normalized_center_error(np.array([box]), np.array([a]))[0])

        ious = np.array(ious)
        norm_center_errors = np.array(norm_center_errors)
        
        return ious, norm_center_errors


    def _calc_curves(self, ious, norm_center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        norm_center_errors = np.asarray(norm_center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_nce = np.linspace(0, 0.5, self.nbins_nce)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_nce = np.less_equal(norm_center_errors, thr_nce)

        succ_curve = np.mean(bin_iou, axis=0)
        norm_prec_curve = np.mean(bin_nce, axis=0)

        return succ_curve, norm_prec_curve


    def _calc_curves_robustness(self, ious):
        seq_length = ious.shape[0]

        thr_iou = np.linspace(0, 0.5, self.nbins_gsr)

        gen_succ_rob_curve = np.zeros(thr_iou.shape[0])
        for i, th in enumerate(thr_iou):
            broken = False
            for j, iou in enumerate(ious):
                if iou <= th:
                    gen_succ_rob_curve[i] = float(j) / seq_length
                    broken = True
                    break
            if not broken:
                gen_succ_rob_curve[i] = 1.0

        return gen_succ_rob_curve