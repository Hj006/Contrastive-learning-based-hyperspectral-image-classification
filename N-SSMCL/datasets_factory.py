#datasets_factory.py
from pathlib import Path
from datasets import load_whu_longkou
from datasets import load_pavia_university_with_full_test  # 你实际文件名
from datasets import load_houston2013                   # 你实际文件名

def get_dataset(name, data_path, **kwargs):
    if name == 'whu':
        return load_whu_longkou(Path(data_path), **kwargs)
    elif name == 'paviau':
        return load_pavia_university_with_full_test(Path(data_path), **kwargs)
    elif name == 'houston2013':
        return load_houston2013(Path(data_path))
    else:
        raise ValueError(f"Unknown dataset: {name}")
