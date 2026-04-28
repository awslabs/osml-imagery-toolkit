#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

import unittest
from pathlib import Path

import aws.osml.formats.sidd.models.sidd_v2_0_0 as sidd2
from aws.osml.formats.model_utils import sidd_parser


class TestSIDDFormats(unittest.TestCase):
    def test_sidd_20(self):
        sidd = sidd_parser.from_path(Path("./test/data/sidd/example.sidd.xml"))

        assert isinstance(sidd, sidd2.SIDD)
