"""Boilerplate code for a progress bar using tqdm or no bar if not installed."""

import warnings


def progress_bar(display, total, description):
    """Get a tqdm progress bar interface.

    If the tqdm library is not installed, this will be an empty progress bar
    that does nothing.

    Args:
        display (bool):
            if true, a real progress bar will be returned.
        total (int):
            the total size of the progress bar.
        description (str):
            description to print in progress bar.
    """
    try:
        # pylint: disable=import-outside-toplevel
        import tqdm
    except ImportError:
        tqdm = None

    if display:
        if tqdm is None:
            warnings.warn(
                "tqdm library needs to be installed to show a " " progress bar."
            )
            return _EmptyBar()

        return tqdm.tqdm(total=total, desc=description)
        # if display is True:
        #   return tqdm.tqdm(total=total, desc=description)
        # else:
        #    return getattr(tqdm, "tqdm_" + display)(total=total)
    return _EmptyBar()


class _EmptyBar:
    """A dummy progress bar.

    Idea take from emce:
    https://github.com/dfm/emcee/blob/main/src/emcee/pbar.py
    """

    # pylint: disable=missing-function-docstring

    def __init__(self):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, *args):
        pass
