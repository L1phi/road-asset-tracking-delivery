import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('delivery_run', Path(__file__).with_name('run.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class PackageTests(unittest.TestCase):
    def test_integrity_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'asset.txt'
            p.write_text('original', encoding='utf-8')
            (root / 'manifest.json').write_text(json.dumps({'files':[
                {'path':'asset.txt', 'sha256':module.sha(p)}]}), encoding='utf-8')
            self.assertEqual(module.verify_package(root), 1)
            p.write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                module.verify_package(root)

    def test_reject_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text(json.dumps({'files':[
                {'path':'../outside', 'sha256':'invalid'}]}), encoding='utf-8')
            with self.assertRaises(ValueError):
                module.verify_package(root)

    def test_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                module.inspect_inputs(Path(tmp), None)

if __name__ == '__main__':
    unittest.main()
