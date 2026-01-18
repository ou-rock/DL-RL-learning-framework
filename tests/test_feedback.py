"""Tests for beta feedback collection"""
import pytest
from pathlib import Path
from learning_framework.feedback import FeedbackCollector, FeedbackType


@pytest.fixture
def collector(tmp_path):
    """Create feedback collector with test database"""
    return FeedbackCollector(db_path=tmp_path / "feedback.db")


def test_submit_bug_report(collector):
    """Can submit bug reports"""
    feedback_id = collector.submit(
        feedback_type=FeedbackType.BUG,
        title="Quiz crashes on large question sets",
        description="When generating quiz with 100+ questions, app crashes",
        steps_to_reproduce="1. Run lf quiz\n2. Select all questions",
        context={"concept": "backpropagation", "num_questions": 100}
    )
    assert feedback_id is not None
    assert feedback_id.startswith("FB-")


def test_submit_feature_request(collector):
    """Can submit feature requests"""
    feedback_id = collector.submit(
        feedback_type=FeedbackType.FEATURE,
        title="Add dark mode for visualizations",
        description="Would love dark theme for late night studying"
    )
    assert feedback_id is not None


def test_submit_usability_feedback(collector):
    """Can submit usability feedback"""
    feedback_id = collector.submit(
        feedback_type=FeedbackType.USABILITY,
        title="Challenge instructions unclear",
        description="backprop_fill challenge needs better examples",
        severity="medium"
    )
    assert feedback_id is not None


def test_list_feedback(collector):
    """Can list submitted feedback"""
    collector.submit(FeedbackType.BUG, "Bug 1", "Description 1")
    collector.submit(FeedbackType.FEATURE, "Feature 1", "Description 2")

    all_feedback = collector.list_feedback()
    assert len(all_feedback) == 2


def test_filter_feedback_by_type(collector):
    """Can filter feedback by type"""
    collector.submit(FeedbackType.BUG, "Bug 1", "Description 1")
    collector.submit(FeedbackType.BUG, "Bug 2", "Description 2")
    collector.submit(FeedbackType.FEATURE, "Feature 1", "Description 3")

    bugs = collector.list_feedback(feedback_type=FeedbackType.BUG)
    assert len(bugs) == 2

    features = collector.list_feedback(feedback_type=FeedbackType.FEATURE)
    assert len(features) == 1


def test_get_feedback_by_id(collector):
    """Can retrieve specific feedback by ID"""
    feedback_id = collector.submit(
        FeedbackType.BUG,
        "Test Bug",
        "Test Description"
    )

    feedback = collector.get_feedback(feedback_id)
    assert feedback is not None
    assert feedback.id == feedback_id
    assert feedback.title == "Test Bug"


def test_export_feedback(collector, tmp_path):
    """Can export feedback to JSON"""
    collector.submit(FeedbackType.BUG, "Test bug", "Description")

    export_path = tmp_path / "feedback_export.json"
    collector.export(export_path)

    assert export_path.exists()

    import json
    with open(export_path, 'r') as f:
        data = json.load(f)
    assert "feedback" in data
    assert len(data["feedback"]) == 1


def test_feedback_includes_system_info(collector):
    """Feedback automatically includes system info"""
    feedback_id = collector.submit(
        FeedbackType.BUG,
        "Test",
        "Description"
    )

    feedback = collector.get_feedback(feedback_id)
    assert feedback.system_info is not None
    assert 'python_version' in feedback.system_info
    assert 'platform' in feedback.system_info
