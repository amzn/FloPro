"""Property-based tests for unique run directories.

**Validates: Requirements 4.3**

This module contains property-based tests that verify multiple
FileSystemBackend instances create unique run directories.
"""

import tempfile
from pathlib import Path

from hypothesis import given, settings, strategies as st

from flo_pro_sdk.core.persistence_backend import FileSystemBackend


class TestUniqueRunDirectories:
    """Property-based tests for unique run directory creation.

    **Validates: Requirements 4.3**
    - 4.3 [Create] Each run creates a unique subdirectory (timestamp or UUID based)
    """

    @given(
        run_count=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=50, deadline=None)
    def test_multiple_stores_create_unique_directories(self, run_count):
        """Property: Multiple FileSystemBackend instances create unique directories.

        **Validates: Requirements 4.3**

        When creating multiple FileSystemBackend instances with the same
        base directory, each should create a unique run subdirectory.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            stores = []
            run_dirs = []

            for _ in range(run_count):
                store = FileSystemBackend(base_dir=tmpdir)
                stores.append(store)
                run_dirs.append(store.run_dir)

            # Verify all run directories are unique
            unique_dirs = set(run_dirs)
            assert len(unique_dirs) == run_count, (
                f"Expected {run_count} unique run directories, "
                f"but got {len(unique_dirs)}. "
                f"Directories: {[str(d) for d in run_dirs]}"
            )

            # Verify all directories actually exist
            for run_dir in run_dirs:
                assert run_dir.exists(), f"Run directory {run_dir} does not exist"
                assert run_dir.is_dir(), f"Run directory {run_dir} is not a directory"

            # Clean up stores
            for store in stores:
                store.close()

    @given(
        run_count=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_unique_run_ids(self, run_count):
        """Property: Multiple FileSystemBackend instances have unique run_ids.

        **Validates: Requirements 4.3**

        When creating multiple FileSystemBackend instances, each should
        have a unique run_id.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            stores = []
            run_ids = []

            for _ in range(run_count):
                store = FileSystemBackend(base_dir=tmpdir)
                stores.append(store)
                run_ids.append(store.run_id)

            # Verify all run_ids are unique
            unique_ids = set(run_ids)
            assert len(unique_ids) == run_count, (
                f"Expected {run_count} unique run_ids, "
                f"but got {len(unique_ids)}. "
                f"Run IDs: {run_ids}"
            )

            # Clean up stores
            for store in stores:
                store.close()

    @given(
        run_count=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_run_directories_under_base_dir(self, run_count):
        """Property: All run directories are created under the base directory.

        **Validates: Requirements 4.3**

        All run directories should be under the base directory
        (base_dir/coordination_id/run_id).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            stores = []

            for _ in range(run_count):
                store = FileSystemBackend(base_dir=tmpdir)
                stores.append(store)

            # Verify all run directories are under base_dir
            for store in stores:
                assert base_path in store.run_dir.parents, (
                    f"Run directory {store.run_dir} is not under "
                    f"base directory {base_path}"
                )

            # Clean up stores
            for store in stores:
                store.close()
