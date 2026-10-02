"""Tests for channel module."""

import numpy as np
import pytest

from dronecomm.channel import ChannelModel, SUBURBAN, URBAN, DENSE_URBAN


def test_los_probability_overhead():
    """At 90 deg elevation (directly overhead), P_LoS should be near 1."""
    ch = ChannelModel(env=URBAN)
    p = ch.los_probability(np.array(np.pi / 2))
    assert p > 0.95


def test_los_probability_low_angle_urban():
    """At very low elevation in urban, P_LoS should be low."""
    ch = ChannelModel(env=URBAN)
    p = ch.los_probability(np.array(np.deg2rad(5.0)))
    assert p < 0.5


def test_los_probability_suburban_higher_than_urban():
    """Suburban should have higher P_LoS than urban at same angle."""
    ch_sub = ChannelModel(env=SUBURBAN)
    ch_urb = ChannelModel(env=URBAN)
    theta = np.array(np.deg2rad(30.0))
    assert ch_sub.los_probability(theta) > ch_urb.los_probability(theta)


def test_fspl_increases_with_distance():
    """Free-space path loss should increase with distance."""
    ch = ChannelModel()
    d = np.array([100.0, 500.0, 1000.0])
    fspl = ch.fspl_db(d)
    assert fspl[0] < fspl[1] < fspl[2]


def test_fspl_known_value():
    """FSPL at 1 km, 2 GHz should be about 106.4 dB."""
    ch = ChannelModel(f_c=2.0e9)
    fspl = ch.fspl_db(np.array(1000.0))
    # 20*log10(4*pi*1000*2e9/3e8) = 20*log10(83776) ≈ 98.5 dB... let me compute
    expected = 20.0 * np.log10(4.0 * np.pi * 1000.0 * 2.0e9 / 3.0e8)
    assert fspl == pytest.approx(expected, rel=1e-6)


def test_path_loss_greater_than_fspl():
    """Total path loss should always be >= FSPL (excess loss is non-negative)."""
    ch = ChannelModel(env=URBAN)
    d = np.array([500.0])
    theta = np.array([np.deg2rad(45.0)])
    pl = ch.path_loss_db(d, theta)
    fspl = ch.fspl_db(d)
    assert pl >= fspl


def test_a2a_equals_fspl():
    """Air-to-air path loss should be pure FSPL."""
    ch = ChannelModel()
    d = np.array([800.0])
    assert ch.a2a_path_loss_db(d) == pytest.approx(ch.fspl_db(d))


def test_path_loss_linear_range():
    """Linear path loss should be between 0 and 1."""
    ch = ChannelModel(env=URBAN)
    d = np.array([100.0, 500.0, 1000.0])
    theta = np.array([np.deg2rad(60.0), np.deg2rad(30.0), np.deg2rad(10.0)])
    pl = ch.path_loss_linear(d, theta)
    assert np.all(pl > 0)
    assert np.all(pl < 1)
