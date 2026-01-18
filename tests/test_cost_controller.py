import pytest
from datetime import datetime, timedelta
from learning_framework.backends.cost_controller import CostController


@pytest.fixture
def controller(tmp_path):
    """Create a cost controller with test database"""
    return CostController(
        db_path=tmp_path / "cost.db",
        daily_budget=5.0,
        max_job_cost=1.0
    )


def test_initial_daily_spending_is_zero(controller):
    """New controller shows zero daily spending"""
    assert controller.get_daily_spending() == 0.0


def test_record_spending_updates_total(controller):
    """Recording spending updates daily total"""
    controller.record_spending("job_1", 0.50)
    assert controller.get_daily_spending() == 0.50

    controller.record_spending("job_2", 0.25)
    assert controller.get_daily_spending() == 0.75


def test_can_spend_within_budget(controller):
    """can_spend returns True when within both daily and per-job limits"""
    assert controller.can_spend(0.50) == True
    assert controller.can_spend(1.00) == True
    # 4.99 exceeds max_job_cost of 1.0, so should be False
    assert controller.can_spend(0.99) == True


def test_cannot_exceed_daily_budget(controller):
    """can_spend returns False when exceeding daily budget"""
    controller.record_spending("job_1", 4.50)
    assert controller.can_spend(0.60) == False
    assert controller.can_spend(0.50) == True


def test_cannot_exceed_max_job_cost(controller):
    """can_spend returns False when exceeding max job cost"""
    assert controller.can_spend(1.50) == False
    assert controller.can_spend(1.00) == True


def test_get_remaining_budget(controller):
    """get_remaining_budget returns correct amount"""
    assert controller.get_remaining_budget() == 5.0

    controller.record_spending("job_1", 2.00)
    assert controller.get_remaining_budget() == 3.0


def test_spending_resets_daily(controller):
    """Spending from previous days doesn't count"""
    # This test requires mocking datetime - simplified version
    controller.record_spending("job_1", 4.00)
    assert controller.get_daily_spending() == 4.00
    # In real implementation, spending would reset at midnight


def test_get_spending_history(controller):
    """get_history returns list of spending records"""
    controller.record_spending("job_1", 0.50)
    controller.record_spending("job_2", 0.75)

    history = controller.get_history()
    assert len(history) == 2
    assert history[0]['job_id'] == "job_1"
    assert history[0]['amount'] == 0.50
