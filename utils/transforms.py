import albumentations as A
import cv2

# ============================================================================
# CNN-Friendly Augmentation (Moderate) — DEFAULT for all architectures
# ============================================================================
# Proven effective on IAM (~6.5K lines). These five transforms cover all the
# meaningful variation in handwritten text: geometry, stroke width, and
# photometric conditions. Every transform has direct HTR-literature backing.
aug_transforms_cnn = A.Compose([

    # Affine: pen tilt (shear), writer size (scale), line shift (translate)
    # Most impactful single transform for HTR — keep probability high.
    A.Affine(
        rotate=(-1, 1),
        shear={'x': (-30, 30), 'y': (-5, 5)},
        scale=(0.6, 1.2),
        translate_percent=0.02,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT,
        p=0.5),

    # Elastic / grid distortion: simulates pen stroke warping and paper curvature
    A.OneOf([
        A.GridDistortion(distort_limit=(-.1, .1), p=0.5),
        A.ElasticTransform(alpha=60, sigma=20, p=0.5),
    ], p=0.5),

    # Morphological: simulates ink thickness (bold vs light strokes)
    A.OneOf([
        A.Morphological(p=0.5, scale=3, operation='dilation'),
        A.Morphological(p=0.5, scale=3, operation='erosion'),
    ], p=0.5),

    # Photometric: scanning condition variation (brightness, contrast)
    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),

    # Gamma: document aging and scanner gamma
    A.RandomGamma(p=0.5, gamma_limit=(80, 120)),
])


# ============================================================================
# ViT-Optimized Augmentation (Strong) — for vit_rgts trained from scratch
# ============================================================================
# Same proven core transforms as CNN aug, with slightly higher probabilities
# and tighter geometric bounds (ViT has no built-in translation invariance).
# Removed: CoarseDropout (destroys characters), Blur/Sharpen (marginal),
#          Perspective (marginal for pre-cropped lines), vit_strong (overkill).
aug_transforms_vit = A.Compose([

    # Affine: wider rotation range for ViT (must learn it explicitly)
    A.Affine(
        rotate=(-10, 10),
        shear={'x': (-35, 35), 'y': (-8, 8)},
        scale=(0.85, 1.15),
        translate_percent=0.03,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT,
        p=0.7),

    # Elastic / grid distortion: same as CNN, slightly stronger
    A.OneOf([
        A.ElasticTransform(alpha=80, sigma=15, p=1.0),
        A.GridDistortion(distort_limit=(-0.15, 0.15), p=1.0),
    ], p=0.8),

    # Morphological: ink thickness variation
    A.OneOf([
        A.Morphological(scale=(2, 4), operation='dilation', p=1.0),
        A.Morphological(scale=(2, 4), operation='erosion', p=1.0),
    ], p=0.6),

    # Photometric: wider range since ViT has no built-in invariance
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.RandomGamma(gamma_limit=(70, 130), p=0.6),

    # GaussNoise: sensor/paper texture noise — ViT benefits more than CNN
    A.GaussNoise(std_range=(0.02, 0.08), p=0.5),
])


# ============================================================================
# Default alias (used by legacy imports)
# ============================================================================
aug_transforms = aug_transforms_cnn
