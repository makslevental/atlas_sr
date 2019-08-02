import unittest

from dsiac.util import *


class TestMethods(unittest.TestCase):
    def test_path(self):
        print(make_dsiac_paths("test"))
        print(make_yuma_paths("test"))


if __name__ == "__main__":
    unittest.main()
