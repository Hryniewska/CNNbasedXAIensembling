import numpy as np
import cv2


def randomHueSaturationValue(
    image,
    hue_shift_limit=(-180, 180),
    sat_shift_limit=(-255, 255),
    val_shift_limit=(-255, 255),
    u=0.5,
):
    """
    Apply random hue, saturation, and value shifts to an input image.

    Args:
        - image: Input image to be transformed.
        - hue_shift_limit: Tuple of minimum and maximum hue shifts.
        - sat_shift_limit: Tuple of minimum and maximum saturation shifts.
        - val_shift_limit: Tuple of minimum and maximum value shifts.
        - u: Probability of applying the transformation.

    Returns:
        Transformed image with random hue, saturation, and value shifts.
    """
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(
    image,
    mask,
    shift_limit=(-0.0, 0.0),
    scale_limit=(-0.0, 0.0),
    rotate_limit=(-0.0, 0.0),
    aspect_limit=(-0.0, 0.0),
    borderMode=cv2.BORDER_CONSTANT,
    u=0.5,
):
    """
    Apply random shift, scale, and rotation transformations to an input image and mask.

    Args:
        - image: Input image to be transformed.
        - mask: Corresponding mask image to be transformed.
        - shift_limit: Tuple of minimum and maximum shift limits.
        - scale_limit: Tuple of minimum and maximum scale limits.
        - rotate_limit: Tuple of minimum and maximum rotation limits.
        - aspect_limit: Tuple of minimum and maximum aspect ratio limits.
        - borderMode: Border mode for image transformation.
        - u: Probability of applying the transformation.

    Returns:
        Transformed image and mask with random shift, scale, and rotation.
    """

    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect**0.5)
        sy = scale / (aspect**0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array(
            [
                [0, 0],
                [width, 0],
                [width, height],
                [0, height],
            ]
        )
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array(
            [width / 2 + dx, height / 2 + dy]
        )

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(
            image,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(
                0,
                0,
                0,
            ),
        )
        mask = cv2.warpPerspective(
            mask,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(
                0,
                0,
                0,
            ),
        )

    return image, mask


def randomFlip(image, mask, u=0.5):
    """
    Randomly flip an input image and its corresponding mask.

    Args:
        - image: Input image to be flipped.
        - mask: Corresponding mask image to be flipped.
        - u: Probability of applying the flip.

    Returns:
        Flipped image and mask.
    """

    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    """
    Randomly rotate an input image and its corresponding mask by 90 degrees.

    Args:
        - image: Input image to be rotated.
        - mask: Corresponding mask image to be rotated.
        - u: Probability of applying the rotation.

    Returns:
        Rotated image and mask.
    """

    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask
