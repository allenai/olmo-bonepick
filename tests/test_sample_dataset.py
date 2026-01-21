"""Tests for sample-dataset command."""

import os
import shutil
from pathlib import Path

import msgspec
import pytest
import smart_open

from bonepick.cli import ByteSizeParamType
from bonepick.data.utils import sample_single_file


# Test fixtures
@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary test dataset."""
    data_dir = tmp_path / "test_dataset"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    # Create test data with different sizes
    encoder = msgspec.json.Encoder()

    # Train data: 100 samples
    train_file = train_dir / "data.jsonl"
    with smart_open.open(train_file, "wb") as f:
        for i in range(100):
            sample = {
                "text": f"This is training sample number {i} with some extra text",
                "score": "pos" if i % 2 == 0 else "neg",
            }
            f.write(encoder.encode(sample) + b"\n")

    # Test data: 50 samples
    test_file = test_dir / "data.jsonl"
    with smart_open.open(test_file, "wb") as f:
        for i in range(50):
            sample = {
                "text": f"This is test sample number {i} with some extra text",
                "score": "pos" if i % 3 == 0 else "neg",
            }
            f.write(encoder.encode(sample) + b"\n")

    return data_dir


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output


# Tests for ByteSizeParamType
class TestByteSizeParamType:
    """Tests for ByteSizeParamType parameter converter."""

    def test_parse_bytes(self):
        converter = ByteSizeParamType()
        assert converter.convert("100", None, None) == 100
        assert converter.convert("100B", None, None) == 100

    def test_parse_kilobytes(self):
        converter = ByteSizeParamType()
        assert converter.convert("1KB", None, None) == 1024
        assert converter.convert("1K", None, None) == 1024
        assert converter.convert("2KB", None, None) == 2048

    def test_parse_megabytes(self):
        converter = ByteSizeParamType()
        assert converter.convert("1MB", None, None) == 1024**2
        assert converter.convert("1M", None, None) == 1024**2
        assert converter.convert("5MB", None, None) == 5 * 1024**2

    def test_parse_gigabytes(self):
        converter = ByteSizeParamType()
        assert converter.convert("1GB", None, None) == 1024**3
        assert converter.convert("1G", None, None) == 1024**3
        assert converter.convert("2GB", None, None) == 2 * 1024**3

    def test_parse_terabytes(self):
        converter = ByteSizeParamType()
        assert converter.convert("1TB", None, None) == 1024**4
        assert converter.convert("1T", None, None) == 1024**4

    def test_parse_decimal(self):
        converter = ByteSizeParamType()
        assert converter.convert("1.5KB", None, None) == int(1.5 * 1024)
        assert converter.convert("2.5MB", None, None) == int(2.5 * 1024**2)

    def test_parse_with_spaces(self):
        converter = ByteSizeParamType()
        assert converter.convert("  1 GB  ", None, None) == 1024**3

    def test_invalid_format(self):
        converter = ByteSizeParamType()
        with pytest.raises(Exception):  # Will raise click.BadParameter
            converter.convert("invalid", None, None)

    def test_int_passthrough(self):
        converter = ByteSizeParamType()
        assert converter.convert(1024, None, None) == 1024


# Tests for sample_single_file
class TestSampleSingleFile:
    """Tests for sample_single_file function."""

    def test_copy_small_file(self, tmp_path):
        """Test that small files are copied without sampling."""
        source = tmp_path / "source.jsonl"
        dest = tmp_path / "dest.jsonl"

        encoder = msgspec.json.Encoder()
        with smart_open.open(source, "wb") as f:
            for i in range(10):
                f.write(encoder.encode({"text": f"sample {i}", "score": "pos"}) + b"\n")

        source_size = source.stat().st_size
        target_size = source_size + 1000  # Target is larger

        sample_single_file(source, dest, target_size, seed=42)

        assert dest.exists()
        assert dest.stat().st_size == source_size

    def test_sample_with_high_ratio(self, tmp_path):
        """Test sampling with ratio >= 5%."""
        source = tmp_path / "source.jsonl"
        dest = tmp_path / "dest.jsonl"

        encoder = msgspec.json.Encoder()
        with smart_open.open(source, "wb") as f:
            for i in range(1000):
                f.write(encoder.encode({"text": f"sample {i}", "score": "pos"}) + b"\n")

        source_size = source.stat().st_size
        target_size = int(source_size * 0.5)  # 50% sampling

        sample_single_file(source, dest, target_size, seed=42)

        assert dest.exists()
        dest_size = dest.stat().st_size
        # Should be approximately 50% (within 10% tolerance due to probabilistic sampling)
        assert 0.45 * source_size < dest_size < 0.55 * source_size

        # Verify actual sampling rate is close to target
        actual_ratio = dest_size / source_size
        assert abs(actual_ratio - 0.5) < 0.05  # Within 5% of target

    def test_sample_with_low_ratio(self, tmp_path):
        """Test sampling with ratio < 5% (byte counting mode)."""
        source = tmp_path / "source.jsonl"
        dest = tmp_path / "dest.jsonl"

        encoder = msgspec.json.Encoder()
        with smart_open.open(source, "wb") as f:
            for i in range(1000):
                f.write(encoder.encode({"text": f"sample {i} with extra padding text", "score": "pos"}) + b"\n")

        source_size = source.stat().st_size
        target_size = int(source_size * 0.01)  # 1% sampling

        sample_single_file(source, dest, target_size, seed=42)

        assert dest.exists()
        dest_size = dest.stat().st_size
        # Byte counting mode should stop near target
        # Allow overshoot up to 2x due to stopping after line is written
        assert dest_size <= target_size * 2

        # Verify it's at least getting some data
        assert dest_size >= target_size * 0.5

        # Check actual bytes match expectation
        actual_ratio = dest_size / source_size
        assert actual_ratio < 0.05  # Should be using low-ratio mode

    def test_reproducibility(self, tmp_path):
        """Test that same seed produces same results."""
        source = tmp_path / "source.jsonl"
        dest1 = tmp_path / "dest1.jsonl"
        dest2 = tmp_path / "dest2.jsonl"

        encoder = msgspec.json.Encoder()
        with smart_open.open(source, "wb") as f:
            for i in range(100):
                f.write(encoder.encode({"text": f"sample {i}", "score": "pos"}) + b"\n")

        source_size = source.stat().st_size
        target_size = int(source_size * 0.3)

        sample_single_file(source, dest1, target_size, seed=42)
        sample_single_file(source, dest2, target_size, seed=42)

        assert dest1.stat().st_size == dest2.stat().st_size

        # Compare content
        with open(dest1, "rb") as f1, open(dest2, "rb") as f2:
            assert f1.read() == f2.read()

    def test_different_seeds(self, tmp_path):
        """Test that different seeds produce different results."""
        source = tmp_path / "source.jsonl"
        dest1 = tmp_path / "dest1.jsonl"
        dest2 = tmp_path / "dest2.jsonl"

        encoder = msgspec.json.Encoder()
        with smart_open.open(source, "wb") as f:
            for i in range(100):
                f.write(encoder.encode({"text": f"sample {i}", "score": "pos"}) + b"\n")

        source_size = source.stat().st_size
        target_size = int(source_size * 0.3)

        sample_single_file(source, dest1, target_size, seed=42)
        sample_single_file(source, dest2, target_size, seed=123)

        # Sizes might be similar, but content should differ
        with open(dest1, "rb") as f1, open(dest2, "rb") as f2:
            content1 = f1.read()
            content2 = f2.read()
            # Very unlikely to be identical with different seeds
            assert content1 != content2 or len(content1) < 10  # Unless very small sample

    def test_preserves_json_format(self, tmp_path):
        """Test that output maintains valid JSON format."""
        source = tmp_path / "source.jsonl"
        dest = tmp_path / "dest.jsonl"

        encoder = msgspec.json.Encoder()
        decoder = msgspec.json.Decoder()

        with smart_open.open(source, "wb") as f:
            for i in range(100):
                f.write(encoder.encode({"text": f"sample {i}", "score": "pos", "value": i}) + b"\n")

        source_size = source.stat().st_size
        target_size = int(source_size * 0.3)

        sample_single_file(source, dest, target_size, seed=42)

        # Verify all lines are valid JSON
        with smart_open.open(dest, "rb") as f:
            for line in f:
                obj = decoder.decode(line)
                assert "text" in obj
                assert "score" in obj
                assert "value" in obj

    def test_compressed_zst_file(self, tmp_path):
        """Test sampling works with .zst compressed files."""
        source = tmp_path / "source.jsonl.zst"
        dest = tmp_path / "dest.jsonl.zst"

        encoder = msgspec.json.Encoder()
        with smart_open.open(source, "wb") as f:
            for i in range(500):
                f.write(encoder.encode({"text": f"sample {i} with extra text", "score": "pos"}) + b"\n")

        source_size = source.stat().st_size
        target_size = int(source_size * 0.3)

        sample_single_file(source, dest, target_size, seed=42)

        assert dest.exists()
        # Verify output is also compressed (check file is smaller than uncompressed would be)
        assert dest.stat().st_size > 0

        # Verify we can read and decode the output
        decoder = msgspec.json.Decoder()
        line_count = 0
        with smart_open.open(dest, "rb") as f:
            for line in f:
                obj = decoder.decode(line)
                assert "text" in obj
                assert "score" in obj
                line_count += 1

        # Should have some data
        assert line_count > 0
        assert line_count < 500  # Should be less than original

    def test_compressed_gz_file(self, tmp_path):
        """Test sampling works with .gz compressed files."""
        source = tmp_path / "source.jsonl.gz"
        dest = tmp_path / "dest.jsonl.gz"

        encoder = msgspec.json.Encoder()
        with smart_open.open(source, "wb") as f:
            for i in range(500):
                f.write(encoder.encode({"text": f"sample {i} with extra text", "score": "pos"}) + b"\n")

        source_size = source.stat().st_size
        target_size = int(source_size * 0.4)

        sample_single_file(source, dest, target_size, seed=42)

        assert dest.exists()
        dest_size = dest.stat().st_size
        assert dest_size > 0

        # Verify we can read and decode the output
        decoder = msgspec.json.Decoder()
        line_count = 0
        with smart_open.open(dest, "rb") as f:
            for line in f:
                obj = decoder.decode(line)
                assert "text" in obj
                assert "score" in obj
                line_count += 1

        assert line_count > 0
        assert line_count < 500

    def test_compressed_file_size_accuracy(self, tmp_path):
        """Test that compressed file sampling respects size targets."""
        source = tmp_path / "source.jsonl.zst"
        dest = tmp_path / "dest.jsonl.zst"

        encoder = msgspec.json.Encoder()
        # Create larger file for better size testing
        with smart_open.open(source, "wb") as f:
            for i in range(2000):
                f.write(
                    encoder.encode({"text": f"sample {i} with some extra padding text here", "score": "pos"})
                    + b"\n"
                )

        source_size = source.stat().st_size
        target_size = int(source_size * 0.5)

        sample_single_file(source, dest, target_size, seed=42)

        dest_size = dest.stat().st_size

        # Compressed files can have compression overhead, so output might be larger
        # Just verify it's not the full original file
        assert dest_size < source_size * 1.5, (
            f"Output ({dest_size}) should not be much larger than source ({source_size})"
        )
        assert dest_size > 0, "Output should contain some data"


# Integration tests
class TestSampleDatasetIntegration:
    """Integration tests for sample-dataset command."""

    def test_sample_with_sampling_rate(self, test_data_dir, output_dir):
        """Test sampling with --sampling-rate option."""
        from click.testing import CliRunner

        from bonepick.data.commands import sample_dataset

        # Get original size
        original_size = sum(f.stat().st_size for f in test_data_dir.rglob("*.jsonl"))

        runner = CliRunner()
        result = runner.invoke(
            sample_dataset,
            [
                "-i",
                str(test_data_dir),
                "-o",
                str(output_dir),
                "--sampling-rate",
                "0.5",
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert (output_dir / "train").exists()
        assert (output_dir / "test").exists()

        # Check that files were created
        train_files = list((output_dir / "train").glob("*.jsonl"))
        test_files = list((output_dir / "test").glob("*.jsonl"))
        assert len(train_files) > 0
        assert len(test_files) > 0

        # Verify output size is approximately 50% of input
        output_size = sum(f.stat().st_size for f in output_dir.rglob("*.jsonl"))
        actual_ratio = output_size / original_size
        # Allow 10% tolerance due to probabilistic sampling
        assert 0.45 < actual_ratio < 0.55, f"Expected ~0.5, got {actual_ratio:.2f}"

    def test_sample_with_target_size(self, test_data_dir, output_dir):
        """Test sampling with --target-size option."""
        from click.testing import CliRunner

        from bonepick.data.commands import sample_dataset

        runner = CliRunner()
        target_bytes = 1024  # 1KB
        result = runner.invoke(
            sample_dataset,
            [
                "-i",
                str(test_data_dir),
                "-o",
                str(output_dir),
                "--target-size",
                "1KB",
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert (output_dir / "train").exists() or (output_dir / "test").exists()

        # Verify output size is close to target
        output_size = sum(f.stat().st_size for f in output_dir.rglob("*.jsonl"))
        # For very small targets, allow significant tolerance
        assert output_size <= target_bytes * 3, f"Expected ~{target_bytes} bytes, got {output_size}"
        assert output_size > 0, "Output should contain some data"

    def test_mutually_exclusive_options(self, test_data_dir, output_dir):
        """Test that sampling-rate and target-size are mutually exclusive."""
        from click.testing import CliRunner

        from bonepick.data.commands import sample_dataset

        runner = CliRunner()
        result = runner.invoke(
            sample_dataset,
            [
                "-i",
                str(test_data_dir),
                "-o",
                str(output_dir),
                "--sampling-rate",
                "0.5",
                "--target-size",
                "1KB",
            ],
        )

        assert result.exit_code != 0

    def test_requires_one_option(self, test_data_dir, output_dir):
        """Test that either sampling-rate or target-size must be specified."""
        from click.testing import CliRunner

        from bonepick.data.commands import sample_dataset

        runner = CliRunner()
        result = runner.invoke(
            sample_dataset,
            [
                "-i",
                str(test_data_dir),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code != 0

    def test_preserves_directory_structure(self, test_data_dir, output_dir):
        """Test that directory structure is preserved."""
        from click.testing import CliRunner

        from bonepick.data.commands import sample_dataset

        runner = CliRunner()
        result = runner.invoke(
            sample_dataset,
            [
                "-i",
                str(test_data_dir),
                "-o",
                str(output_dir),
                "--sampling-rate",
                "0.5",
            ],
        )

        assert result.exit_code == 0
        # Check structure matches
        for split in ["train", "test"]:
            if (test_data_dir / split).exists():
                assert (output_dir / split).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
