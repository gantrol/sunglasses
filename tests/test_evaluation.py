"""Synthetic evaluation controls only: no windows, capture, GPU or playback."""

import json
import contextlib
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from sunglasses.evaluation.evidence import new_directory, write_json
from sunglasses.evaluation.__main__ import main
from sunglasses.evaluation.metrics import excursions, paired, sampling
from sunglasses.evaluation.recording import findings, telemetry, white_intervals
from sunglasses.evaluation.suite import run_suite
from sunglasses.presentation_scenes import Profile, Scene
from sunglasses.presentation_sim import Timing, simulate
from tests.support import workspace_temp


class MeasurementTests(unittest.TestCase):
    def test_identity_has_events_but_no_benefit(self):
        source = np.array([.02, 1, 1, .02])
        result = excursions(source, source, 100)
        self.assertEqual(result['event_count'], 1)
        self.assertEqual(result['events_with_reduction'], 0)
        event = result['events'][0]
        self.assertEqual(event['peak_reduction_fraction'], 0)
        self.assertEqual(event['excess_reduction_fraction'], 0)
        self.assertIsNone(event['first_reduction_seconds'])
        self.assertEqual(event['unreduced_seconds'], .02)

    def test_late_dimming_does_not_hide_leaked_peak(self):
        source = np.array([.02, 1, 1, 1, .02])
        result = excursions(source, [.02, 1, .6, .6, .02], 100)
        event = result['events'][0]
        self.assertEqual(event['peak_reduction_fraction'], 0)
        self.assertEqual(event['initial_unreduced_samples'], 1)
        self.assertEqual(event['first_reduction_seconds'], .01)
        self.assertGreater(event['excess_reduction_fraction'], 0)
        expected = ((1 - .51) + 2 * (.6 - .51)) / 100
        self.assertAlmostEqual(event['output_excess_integral'], expected)

    def test_output_cannot_change_source_event_selection(self):
        source = [.01, .2, .2, .01]
        identity, dim = excursions(source, source, 60), excursions(source, [0, 0, 0, 0], 60)
        self.assertEqual(identity['source_threshold'], dim['source_threshold'])
        self.assertEqual(identity['event_count'], dim['event_count'])
        self.assertEqual(dim['events_with_reduction'], 1)
        self.assertEqual(dim['events'][0]['excess_reduction_fraction'], 1)

    def test_constant_source_does_not_invent_flash_benefit(self):
        result = excursions([1, 1, 1], [.5, .5, .5], 60)
        self.assertEqual(result['event_count'], 0)

    def test_censored_bright_interval_is_marked(self):
        result = excursions([.01, 1, 1], [.01, .6, .6], 60)
        self.assertTrue(result['events'][0]['right_censored'])

    def test_sampling_uses_actual_pts_and_rejects_duplicates(self):
        result = sampling([0, .008, .017, .042], 120)
        self.assertEqual(result['gaps_over_1_5_periods'], 1)
        self.assertAlmostEqual(result['maximum_gap_seconds'], .025)
        self.assertAlmostEqual(result['observed_mean_hz'], 3 / .042)
        for times in ([0], [0, 0], [1, 0], [0, float('nan')]):
            with self.assertRaises(ValueError):
                sampling(times)

    def test_dark_detail_damage_is_not_hidden_in_roi_mean(self):
        source = np.full((3, 2, 2, 3), 255, np.uint8)
        source[:, 0, 0] = 40
        output = source.copy()
        output[:, 0, 0] = 20
        scene = Scene('detail', ('S4',), .05, lambda t: source[0],
                      {'full': np.ones((2, 2), bool)}, unchanged=('full',))
        report, traces = paired(scene, source, output, 60)
        self.assertEqual(report['dark_pixel_changed_fraction'], 1)
        self.assertEqual(report['dark_pixel_max_code_change'], 20)
        self.assertEqual(report['changed_pixel_fraction'], .25)
        self.assertTrue(report['rois']['full']['source_pixels_unchanged'])
        self.assertEqual(len(traces['full__source']), 3)

    def test_unaligned_shapes_and_invalid_luminance_are_rejected(self):
        with self.assertRaises(ValueError):
            excursions([1, 0], [1], 60)
        with self.assertRaises(ValueError):
            excursions([1, -1], [1, 0], 60)


class RecordingTests(unittest.TestCase):
    def metadata(self, mode='visible'):
        return {'sequence_completed': True, 'recording_mode': mode,
                'holds': [{'samples': 10, 'capture_mean_min': .999, 'capture_mean_max': .999,
                           'repeated_triggers': 0} for _ in range(2)]}

    def test_hidden_recording_cannot_prove_output_effect(self):
        result = findings(self.metadata('hidden'))
        self.assertEqual(result['output_effect'], 'not_observable_in_hidden_recording')
        self.assertEqual(result['overall_effectiveness'], 'not_assessed')

    def test_sparse_video_retains_timestamps_and_separates_black_edges(self):
        times = np.array([0, .01, .03, .52, .53, .54, .56, 1.07, 1.08, 1.09])
        values = np.array([0, 255, 196, 255, 255, 0, 255, 229, 255, 0])
        intervals = white_intervals(times, values)
        self.assertEqual(len(intervals), 2)
        self.assertAlmostEqual(intervals[0]['first_dim_seconds'], .02)
        self.assertEqual(intervals[0]['minimum_code'], 196)
        self.assertEqual(intervals[0]['steady_peak_to_peak'], 0)
        result = findings(self.metadata(), intervals)
        self.assertEqual(result['output_effect'], 'dimming_observed')
        self.assertEqual(result['output_stability'], 'settled_variation_observed')

    def test_aborted_missing_and_censored_evidence_cannot_pass(self):
        metadata = self.metadata()
        metadata['sequence_completed'] = False
        self.assertEqual(findings(metadata)['capture_stability'], 'insufficient_evidence')
        metadata = self.metadata()
        metadata['holds'][0]['samples'] = 0
        self.assertEqual(findings(metadata)['capture_stability'], 'insufficient_evidence')
        intervals = white_intervals(np.array([0, .5, 1.]), np.array([255, 196, 255]))
        self.assertTrue(intervals[0]['left_censored'])
        self.assertEqual(findings(self.metadata(), intervals)['output_effect'], 'insufficient_evidence')

    def test_telemetry_bounds_deduplication_and_strength_pairing(self):
        timeline = {'qpc_frequency': 100, 'recording_mode': 'visible', 'sequence_completed': True,
                    'events': [{'qpc': 1000, 'label': 'begin'}, {'qpc': 1010, 'label': 'strength 100'},
                               {'qpc': 1020, 'label': 'white'}, {'qpc': 1200, 'label': 'black'}]}
        row = {'type': 'gpu', 'qpc': 1100, 'refresh': False, 'mean_luminance': .99, 'triggered': False}
        records = [row, row, {**row, 'qpc': 1, 'mean_luminance': .1}, {**row, 'qpc': 1150, 'refresh': True}]
        result = telemetry(timeline, records)
        self.assertEqual(result['holds'][0]['samples'], 1)
        self.assertEqual(result['holds'][0]['strength_percent'], 100)
        self.assertEqual(result['holds'][0]['capture_mean_min'], .99)


class ModelAndEvidenceTests(unittest.TestCase):
    def scene(self):
        return Scene('fixture', ('S1',), 1., lambda t: np.full((2, 2, 3), 255, np.uint8),
                     {'full': np.ones((2, 2), bool)}, unchanged=('full',))

    def test_periodic_capture_is_an_explicit_model_and_off_is_identity(self):
        scene = self.scene()
        source, clean, clean_log = simulate(scene, 120, Timing('clean', 2), .5)
        _, periodic, periodic_log = simulate(scene, 120, Timing('periodic', 2, capture_hz=60), .5)
        self.assertEqual(clean_log['captured'].sum(), 1)
        self.assertEqual(periodic_log['captured'].sum(), 60)
        np.testing.assert_array_equal(source, clean)
        np.testing.assert_array_equal(source, periodic)
        with self.assertRaises(ValueError):
            simulate(scene, 60, Timing('invalid', 2, capture_hz=120), .5)

    def test_existing_or_original_evidence_is_never_overwritten(self):
        with workspace_temp() as folder:
            source = Path(folder) / 'original'
            source.mkdir()
            with self.assertRaises(FileExistsError):
                new_directory(source)
            with self.assertRaises(ValueError):
                new_directory(source / 'derived', source=source)
            with self.assertRaises(ValueError):
                write_json(Path(folder) / 'nonfinite.json', {'value': float('nan')})

    def test_duplicate_selection_is_rejected_before_writing(self):
        with workspace_temp() as folder:
            out = Path(folder) / 'duplicate'
            with self.assertRaises(ValueError):
                run_suite(out, ('portrait', 'portrait'))
            self.assertFalse(out.exists())

    def test_decoder_failure_has_the_documented_error_exit(self):
        with patch('sunglasses.evaluation.__main__.analyze_run', side_effect=subprocess.CalledProcessError(1, 'ffmpeg')), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(['recording', 'unused-input', '--out', 'unused-output'])
        self.assertEqual(caught.exception.code, 2)

    def test_runner_produces_identical_input_three_way_comparison(self):
        profile = Profile('synthetic', (2, 2), 60., (2, 2))
        with workspace_temp() as folder, patch('sunglasses.evaluation.suite.PROFILES', {'fixture': profile}), \
                patch('sunglasses.evaluation.suite.build', return_value=[self.scene()]), \
                patch('sunglasses.evaluation.suite.contract_suite.self_check', return_value=[{'ok': True}]):
            out = Path(folder) / 'comparison'
            report = run_suite(out, ('fixture',), ('clean-lag2',), ('fixture',))
            self.assertEqual([row['strength'] for row in report['reports']], [0, .5, 1])
            self.assertEqual(len({row['source_frames_sha256'] for row in report['reports']}), 1)
            self.assertTrue(all(row['measurements']['identity'] for row in report['reports']))
            self.assertEqual(json.loads((out / 'summary.json').read_text())['kind'], 'synthetic-comparison')
            self.assertEqual(len(list((out / 'traces').glob('*.npz'))), 3)
