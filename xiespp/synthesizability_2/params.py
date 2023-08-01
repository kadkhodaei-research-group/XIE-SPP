IMAGE_PARAMS = dict(
    box_size=70-20,  # 50 Angstroms
    n_bins=128,
    channels=['atomic_number', 'group', 'period'],
    filling='fill-cut',
)

# DEFAULT_BOX = CVR.BoxImage(box_size=DEFAULT_IMAGE_PARAMS['box_size'], n_bins=DEFAULT_IMAGE_PARAMS['n_bins'])
