"""Unit checks for curriculum scheduling and checkpoint selection helpers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_visual import (
    get_checkpoint_paths,
    get_scale_curriculum_stage,
    robust_validation_score,
)


def test_curriculum_boundaries():
    assert get_scale_curriculum_stage(0.0)[0] == 'foundation'
    assert get_scale_curriculum_stage(0.299)[0] == 'foundation'
    assert get_scale_curriculum_stage(0.3)[0] == 'transition'
    assert get_scale_curriculum_stage(0.699)[0] == 'transition'
    assert get_scale_curriculum_stage(0.7)[0] == 'robustness'
    assert get_scale_curriculum_stage(1.0)[0] == 'robustness'


def test_checkpoint_paths():
    paths = get_checkpoint_paths('saved_models/run/seed2_best.pth')
    assert paths == {
        'clean_best': 'saved_models/run/seed2_best.pth',
        'robust_best': 'saved_models/run/seed2_robust_best.pth',
        'final': 'saved_models/run/seed2_final.pth',
    }


def test_robust_score_prioritizes_worst_scale():
    balanced = [{'success_rate': value} for value in [75, 74, 73, 72, 71]]
    brittle = [{'success_rate': value} for value in [95, 95, 95, 95, 50]]
    balanced_score = robust_validation_score(balanced)
    brittle_score = robust_validation_score(brittle)
    assert balanced_score == (71, 73.0)
    assert brittle_score == (50, 86.0)
    assert balanced_score > brittle_score


if __name__ == '__main__':
    test_curriculum_boundaries()
    test_checkpoint_paths()
    test_robust_score_prioritizes_worst_scale()
    print('train_visual helper tests passed')
