from combine_data import MAX_F1, can_reach_f1


def test_equal_sizes_can_reach_f1():
    assert can_reach_f1(100, 100)


def test_size_inside_band_can_reach_f1():
    assert can_reach_f1(100, 82)
    assert can_reach_f1(82, 100)
    assert can_reach_f1(100, 122)


def test_size_outside_band_cannot_reach_f1():
    assert not can_reach_f1(100, 81)
    assert not can_reach_f1(81, 100)
    assert not can_reach_f1(100, 123)


def test_zero_size_cannot_reach_f1():
    assert not can_reach_f1(0, 100)
    assert not can_reach_f1(100, 0)


def test_band_boundaries_match_f1_threshold():
    a_size, b_size = 100, 82
    assert 2.0 * min(a_size, b_size) / (a_size + b_size) >= MAX_F1
    assert can_reach_f1(a_size, b_size)
    b_size = 81
    assert 2.0 * min(a_size, b_size) / (a_size + b_size) < MAX_F1
    assert not can_reach_f1(a_size, b_size)
