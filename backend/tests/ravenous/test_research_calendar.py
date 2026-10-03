"""Relative periods share one calendar reference across month and year boundaries."""

import datetime as dt

import pytest
from open_webui.ravenous_research import evidence


@pytest.mark.parametrize(
    'day,first,last',
    [
        (dt.date(2026, 10, 3), '2026-09-28', '2026-10-04'),
        (dt.date(2027, 1, 1), '2026-12-28', '2027-01-03'),
        (dt.date(2028, 2, 29), '2028-02-28', '2028-03-05'),
    ],
)
def test_week_reference_includes_every_day_and_preserves_the_weekend(day, first, last):
    calendar = evidence.calendar_context(day)
    dates = calendar['week_dates']
    assert len(dates) == 7 and dates[0] == first and dates[-1] == last
    assert calendar['utc_date'] == day.isoformat()
    assert calendar['weekend_dates'] == dates[-3:] == evidence.weekend_dates(day)
    assert [calendar['weekdays'][date] for date in dates] == [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
    ]
    for offset, date in enumerate(dates):
        assert dt.date.fromisoformat(date) == dt.date.fromisoformat(first) + dt.timedelta(days=offset)
        assert evidence.calendar_context(dt.date.fromisoformat(date))['week_dates'] == dates


def test_synthesis_reuses_the_assessed_calendar_after_the_clock_crosses_into_a_new_week():
    prompt = evidence.context_message([], 'Compare these options', {'utc_date': '2027-01-03'})['content']
    assert '"utc_date": "2027-01-03"' in prompt
    assert '"week_dates": ["2026-12-28"' in prompt
    assert '"2027-01-03": "Sunday"' in prompt
