#  Copyright 2024 Amazon.com, Inc. or its affiliates.

"""Smoke test for the ``examples/build_pyramid.py`` command-line script.

Validates that the script imports cleanly inside the project's tox-conda
environment and that ``--help`` returns a zero exit code. Full end-to-end
round-trip behaviour is covered by
``test/unit/aws/osml/image_processing/test_pyramid_integration.py``.
"""

import os
import runpy
import subprocess
import sys

import pytest

# Absolute path to the example script.
_EXAMPLES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples"))
_SCRIPT_PATH = os.path.join(_EXAMPLES_DIR, "build_pyramid.py")


@pytest.mark.skipif(not os.path.isfile(_SCRIPT_PATH), reason="build_pyramid.py example not present")
def test_script_help_exits_zero():
    """The script's ``--help`` output parses and exits with status 0."""
    result = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Build a COG or NITF R-Set pyramid" in result.stdout
    assert "--format" in result.stdout


@pytest.mark.skipif(not os.path.isfile(_SCRIPT_PATH), reason="build_pyramid.py example not present")
def test_script_roundtrip_cog(tmp_path):
    """Invoke the script against a real TIFF fixture and verify COG output.

    Mirrors the end-to-end round-trip validated by the integration test
    suite but routes through the CLI entry point so we exercise the
    argparse + ``main()`` path in ``examples/build_pyramid.py`` as a
    user would from the shell.
    """
    source = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "test", "data", "small.tif"))
    if not os.path.isfile(source):
        pytest.skip("test/data/small.tif not available")

    output = tmp_path / "out.tif"
    # Use runpy so coverage still sees the example module and any
    # import-time failures surface as ImportError rather than a
    # subprocess exit code.
    old_argv = sys.argv
    try:
        sys.argv = [
            _SCRIPT_PATH,
            source,
            "--output",
            str(output),
            "--format",
            "cog",
            "--num-workers",
            "0",
        ]
        runpy.run_path(_SCRIPT_PATH, run_name="__main__")
    except SystemExit as exc:
        assert exc.code in (0, None), f"script exited with {exc.code}"
    finally:
        sys.argv = old_argv

    assert output.is_file(), f"script did not produce {output}"
    assert output.stat().st_size > 0
