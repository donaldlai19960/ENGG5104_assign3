import numpy as np

def get_interest_points(image, feature_width):
    """ Returns a set of interest points for the input image
    Args:
        image - can be grayscale or color, your choice.
        feature_width - in pixels, is the local feature width. It might be
            useful in this function in order to (a) suppress boundary interest
            points (where a feature wouldn't fit entirely in the image)
            or (b) scale the image filters being used. Or you can ignore it.
    Returns:
        x and y: nx1 vectors of x and y coordinates of interest points.
        confidence: an nx1 vector indicating the strength of the interest
            point. You might use this later or not.
        scale and orientation: are nx1 vectors indicating the scale and
            orientation of each interest point. These are OPTIONAL. By default you
            do not need to make scale and orientation invariant local features. 
    """
    h, w = image.shape[:2]

    # Placeholder that you can delete -- these are just random points
    x = np.ceil(np.random.rand(500, 1) * w)
    y = np.ceil(np.random.rand(500, 1) * h)


    # If you do not use (confidence, scale, orientation), just delete
    # return x, y, confidence, scale, orientation
    return x, y


def get_features(image, x, y, feature_width):
    """ Returns a set of feature descriptors for a given set of interest points. 
    Args:
        image - can be grayscale or color, your choice.
        x and y: nx1 vectors of x and y coordinates of interest points.
            The local features should be centered at x and y.
        feature_width - in pixels, is the local feature width. You can assume
            that feature_width will be a multiple of 4 (i.e. every cell of your
            local SIFT-like feature will have an integer width and height).
        If you want to detect and describe features at multiple scales or
            particular orientations you can add other input arguments.
    Returns:
        features: the array of computed features. It should have the
            following size: [length(x) x feature dimensionality] (e.g. 128 for
            standard SIFT)
    """
    # Placeholder that you can delete. Empty features.
    features = np.zeros((x.shape[0], 128))

    return features


def match_features(features1, features2, threshold=0.0):
    """ 
    Args:
        features1 and features2: the n x feature dimensionality features
            from the two images.
        threshold: a threshold value to decide what is a good match. This value 
            needs to be tuned.
        If you want to include geometric verification in this stage, you can add
            the x and y locations of the features as additional inputs.
    Returns:
        matches: a k x 2 matrix, where k is the number of matches. The first
            column is an index in features1, the second column is an index
            in features2. 
        Confidences: a k x 1 matrix with a real valued confidence for every
            match.
        matches' and 'confidences' can be empty, e.g. 0x2 and 0x1.
    """

    # Placeholder that you can delete. Random matches and confidences
    num_features = min(features1.shape[0], features2.shape[0])
    matched = np.zeros((num_features, 2))
    matched[:, 0] = np.random.permutation(num_features)
    matched[:, 1] = np.random.permutation(num_features)
    confidence = np.random.rand(num_features, 1)

    # Sort the matches so that the most confident onces are at the top of the
    # list. You should probably not delete this, so that the evaluation
    # functions can be run on the top matches easily.
    order = np.argsort(confidence, axis=0)[::-1, 0]
    confidence = confidence[order, :]
    matched = matched[order, :]

    return matched, confidence