import os
from functools import cache

from pytorch_fid.fid_score import calculate_frechet_distance, compute_statistics_of_path
from pytorch_fid.inception import InceptionV3


@cache
def _load_model(device, dims):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    return model


def compute_fid(
    paths,
    batch_size=256,
    device="cuda",
    dims=2048,
    num_workers=1,
):
    """Calculates the FID of two paths"""
    for p in paths:
        if not p.exists():
            raise RuntimeError("Invalid path: %s" % p)

    paths = [*map(str, paths)]

    model = _load_model(device, dims)

    m1, s1 = compute_statistics_of_path(
        paths[0],
        model,
        batch_size,
        dims,
        device,
        num_workers,
    )

    m2, s2 = compute_statistics_of_path(
        paths[1],
        model,
        batch_size,
        dims,
        device,
        num_workers,
    )

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value
