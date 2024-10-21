import platform

import matplotlib as mpl


def auto_select_mpl_backend():
    """
    Select the appropriate backend for matplotlib according to the current OS.
    ONLY BEING TESTED ON WINDOWS.
    """
    current_os = platform.system()
    if current_os == 'Windows':
        mpl.use('Qt5Agg')
    elif current_os == 'Linux':
        mpl.use('TkAgg')
    elif current_os == 'Darwin':
        mpl.use('MacOSX')
    else:
        raise RuntimeError(f"Unknown OS: {current_os}")
