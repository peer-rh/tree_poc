from continuation_tree.flex import PackedFrontier, build_attention_mask_matrix


def test_mask_blocks_siblings_and_cross_tree_attention():
    frontier = PackedFrontier.from_rows(
        [
            (0, 0, None, 0, 2),
            (0, 1, 0, 1, 3),
            (0, 2, 0, 1, 3),
            (0, 4, 1, 2, 4),
            (1, 0, None, 0, 1),
            (1, 1, 4, 1, 2),
        ]
    )
    mask = build_attention_mask_matrix(frontier)
    assert mask[3][0]
    assert mask[3][1]
    assert not mask[3][2]
    assert not mask[3][4]
    assert mask[5][4]
    assert not mask[5][0]
