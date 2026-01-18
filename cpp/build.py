"""Build C++ extensions"""
import subprocess
import sys
from pathlib import Path


def build_cpp():
    cpp_dir = Path(__file__).parent
    build_dir = cpp_dir / 'build'
    build_dir.mkdir(exist_ok=True)

    print("Configuring CMake...")
    subprocess.run(
        ['cmake', '-B', str(build_dir), '-S', str(cpp_dir)],
        check=True
    )

    print("Building...")
    subprocess.run(
        ['cmake', '--build', str(build_dir), '--config', 'Release'],
        check=True
    )

    print("C++ module built successfully!")

    # Copy to package directory
    import shutil
    import glob

    # Find the built module
    patterns = [
        str(build_dir / '**' / 'learning_core_cpp*.pyd'),  # Windows
        str(build_dir / '**' / 'learning_core_cpp*.so'),   # Linux/Mac
    ]

    for pattern in patterns:
        for src in glob.glob(pattern, recursive=True):
            dst = cpp_dir.parent / 'learning_framework' / Path(src).name
            shutil.copy(src, dst)
            print(f"Copied {src} -> {dst}")
            return

    print("Warning: Could not find built module to copy")


if __name__ == '__main__':
    build_cpp()
