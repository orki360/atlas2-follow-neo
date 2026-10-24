"""Portable GMC experiment settings; independent of GUI and OpenCV."""

GMC_RESOLUTIONS = ('480x270', '640x360', '960x540', '1280x720', '1920x1080')
GMC_DEFAULT_RESOLUTION = '480x270'
GMC_DEFAULT_CORNERS = 400
GMC_MIN_CORNERS = 100
GMC_MAX_CORNERS = 1000


def validate_gmc(resolution, corners):
    if resolution not in GMC_RESOLUTIONS:
        raise ValueError('GMC resolution must be one of: ' + ', '.join(GMC_RESOLUTIONS))
    if type(corners) is not int or not GMC_MIN_CORNERS <= corners <= GMC_MAX_CORNERS:
        raise ValueError(f'GMC maximum points must be an integer from {GMC_MIN_CORNERS} to {GMC_MAX_CORNERS}')
