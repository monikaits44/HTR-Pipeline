import albumentations as A
import cv2

# ============================================================================
# CNN-Friendly Augmentation (Moderate)
# ============================================================================
# Best for CNN+RNN architectures with built-in translation/rotation invariance
aug_transforms_cnn = A.Compose([
    
    # geometric augmentation
    A.Affine(
        rotate=(-1, 1),
        shear={'x': (-30, 30), 'y': (-5, 5)},
        scale=(0.6, 1.2),
        translate_percent=0.02,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT,
        p=0.5),

    # distortions
    A.OneOf([
        A.GridDistortion(distort_limit=(-.1, .1), p=0.5),
        A.ElasticTransform(alpha=60, sigma=20, p=0.5),
    ], p=0.5),

    # erosion & dilation
    A.OneOf([
        A.Morphological(p=0.5, scale=3, operation='dilation'),
        A.Morphological(p=0.5, scale=3, operation='erosion'),
    ], p=0.5),

    # color augmentation - only grayscale images
    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
    
    # color contrast
    A.RandomGamma(p=0.5, gamma_limit=(80, 120)),
])


# ============================================================================
# ViT-Optimized Augmentation (Strong & Diverse)
# ============================================================================
# ViTs have LOW inductive bias and need STRONG augmentation to learn invariances
# Reference: "Vision Transformers Need Registers" + "An Image is Worth 16x16 Words"
aug_transforms_vit = A.Compose([
    
    # ✅ GEOMETRIC AUGMENTATIONS (More aggressive for ViT)
    # ViTs don't assume spatial locality - need to learn rotation/scale invariance
    A.Affine(
        rotate=(-10, 10),              # Increased from ±1° to ±10°
        shear={'x': (-35, 35), 'y': (-8, 8)},  # Slightly increased shear
        scale=(0.85, 1.15),            # Tighter scale range (was 0.6-1.2)
        translate_percent=0.03,        # Slight translation
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT,
        p=0.7),                        # Increased probability
    
    # ✅ PERSPECTIVE TRANSFORM (Simulates viewing angle changes)
    A.Perspective(scale=(0.02, 0.08), p=0.4),
    
    # ✅ HANDWRITING-SPECIFIC DISTORTIONS (Critical for HTR)
    # Elastic transforms simulate pen pressure variations
    A.OneOf([
        A.ElasticTransform(
            alpha=80,                  # Increased distortion strength
            sigma=15,                  # Smoother distortion
            p=1.0
        ),
        A.GridDistortion(
            distort_limit=(-0.15, 0.15),  # Increased distortion
            p=1.0
        ),
    ], p=0.8),                         # Increased probability
    
    # ✅ MORPHOLOGICAL OPERATIONS (Simulate ink thickness variations)
    A.OneOf([
        A.Morphological(scale=(2, 4), operation='dilation', p=1.0),
        A.Morphological(scale=(2, 4), operation='erosion', p=1.0),
    ], p=0.6),
    
    # ✅ PHOTOMETRIC AUGMENTATIONS (Learn brightness/contrast invariance)
    # ViTs need to learn these - no built-in assumptions
    A.RandomBrightnessContrast(
        brightness_limit=0.3,          # Increased from 0.2
        contrast_limit=0.3,            # Increased from 0.2
        p=0.7
    ),
    
    # ✅ GAMMA CORRECTION (Simulates different scanning conditions)
    A.RandomGamma(
        gamma_limit=(70, 130),         # Wider range
        p=0.6
    ),
    
    # ✅ GAUSSIAN NOISE (Robustness to sensor noise)
    A.GaussNoise(
        std_range=(0.02, 0.08),  # Normalized std (5-20 pixel values for 0-255 range)
        p=0.5
    ),
    
    # ✅ RANDOM ERASING (Regularization - simulate occlusions)
    # Small regions only - don't destroy too much text
    A.CoarseDropout(
        num_holes_range=(1, 3),    # Up to 3 small holes
        hole_height_range=(4, 8),  # Small height in pixels
        hole_width_range=(8, 16),  # Small width in pixels
        fill=255,                  # Fill with white (background)
        p=0.4
    ),
    
    # ✅ BLUR (Simulate out-of-focus images)
    A.OneOf([
        A.GaussianBlur(blur_limit=(1, 3), p=1.0),
        A.MotionBlur(blur_limit=(3, 5), p=1.0),
    ], p=0.3),
    
    # ✅ SHARPEN (Compensate for blur, adds variety)
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.3),
])


# ============================================================================
# ViT-Optimized Augmentation (EXTRA STRONG for very large models)
# ============================================================================
# Use this for TrOCR or very large ViT models (>100M params)
# Includes all previous augmentations + additional regularization
aug_transforms_vit_strong = A.Compose([
    
    # Even more aggressive geometric
    A.Affine(
        rotate=(-15, 15),              # ±15° rotation
        shear={'x': (-40, 40), 'y': (-10, 10)},
        scale=(0.8, 1.2),
        translate_percent=0.05,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT,
        p=0.8),
    
    A.Perspective(scale=(0.02, 0.1), p=0.5),
    
    # Strong distortions
    A.OneOf([
        A.ElasticTransform(alpha=100, sigma=12, p=1.0),
        A.GridDistortion(distort_limit=(-0.2, 0.2), p=1.0),
    ], p=0.9),
    
    # Morphological
    A.OneOf([
        A.Morphological(scale=(2, 5), operation='dilation', p=1.0),
        A.Morphological(scale=(2, 5), operation='erosion', p=1.0),
    ], p=0.7),
    
    # Strong photometric
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
    A.RandomGamma(gamma_limit=(60, 140), p=0.7),
    A.GaussNoise(std_range=(0.02, 0.1), p=0.6),  # Normalized std (stronger noise)
    
    # More aggressive erasing
    A.CoarseDropout(
        num_holes_range=(1, 5),
        hole_height_range=(4, 10),
        hole_width_range=(8, 20),
        fill=255,
        p=0.5
    ),
    
    A.OneOf([
        A.GaussianBlur(blur_limit=(1, 5), p=1.0),
        A.MotionBlur(blur_limit=(3, 7), p=1.0),
    ], p=0.4),
    
    A.Sharpen(alpha=(0.1, 0.4), lightness=(0.7, 1.3), p=0.4),
])


# ============================================================================
# Default export (EXTRA STRONG augmentation for all models)
# ============================================================================
# Using the strongest augmentation uniformly across all models for:
# - Better generalization
# - Consistent training across experiments
# - Maximum robustness to variations
aug_transforms = aug_transforms_vit_strong  # Default to EXTRA STRONG for all models
