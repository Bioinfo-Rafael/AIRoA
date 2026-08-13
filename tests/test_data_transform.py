import torch

from airoa.data.libero_plus import PhotometricConfig, Track1ImageTransform


def test_128_bottleneck_restores_shape_and_correlates_cameras():
    image = torch.linspace(0, 1, 3 * 256 * 256).reshape(3, 256, 256)
    transform = Track1ImageTransform(
        bottleneck_size=128,
        restore_feature_resolution=True,
        photometric=PhotometricConfig(enabled=True),
    )
    result = transform(
        {"observation.images.front": image, "observation.images.wrist": image.clone()}
    )
    assert result["observation.images.front"].shape == (3, 256, 256)
    assert torch.equal(result["observation.images.front"], result["observation.images.wrist"])
