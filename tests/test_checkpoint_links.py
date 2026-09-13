import errno
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('links', Path(__file__).resolve().parents[1]/'files/fp8dense/checkpoint_links.py')
links=importlib.util.module_from_spec(spec);spec.loader.exec_module(links)


class CheckpointLinkTests(unittest.TestCase):
    def test_fallback_survives_different_cache_mount_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'cache';source=root/'source'/'blob';target=root/'derived'/'snapshot'/'weight'
            source.parent.mkdir(parents=True);target.parent.mkdir(parents=True)
            source.write_bytes(b'original immutable weights')
            with patch.object(links.os, 'link', side_effect=OSError(errno.EXDEV, 'cross-device')):
                links.link_unchanged(str(source),str(target))
            self.assertFalse(Path(links.os.readlink(target)).is_absolute())
            moved=Path(tmp)/'worker-mount';root.rename(moved)
            self.assertEqual((moved/'derived'/'snapshot'/'weight').read_bytes(), b'original immutable weights')
