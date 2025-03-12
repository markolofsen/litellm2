#!/usr/bin/env python3
import subprocess
from pathlib import Path
from typing import Optional

import questionary


class DeployManager:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.package_name = self._get_package_name()
        self.current_version = self._get_current_version()
        self.style = questionary.Style(
            [
                ("question", "fg:cyan bold"),
                ("answer", "fg:yellow"),
                ("pointer", "fg:cyan bold"),
                ("highlighted", "fg:cyan bold"),
                ("selected", "fg:green bold"),
                ("separator", "fg:blue"),
                ("instruction", "fg:white italic"),
                ("input", "fg:white"),
                ("error", "fg:red bold"),
            ]
        )

        # Defining command groups
        self.commands = {
            "Development": {
                "Check Environment": self.check_environment,
                "Install Package": self.install_package,
                "Run Tests": self.run_tests,
                "Format Code": self.format_code,
                "Check Linting": self.check_lint,
            },
            "Version Control": {
                "Update Version": self.update_version,
                "Show Changes": self.show_changes,
            },
            "Package Management": {
                "Build Package": self.build_package,
                f"Upload to TestPyPI (v{self.current_version})":
                    self.upload_to_testpypi,
                f"Upload to PyPI (v{self.current_version})":
                    self.upload_to_pypi,
                "Clean Builds": self.clean_builds,
            },
        }

    def _get_package_name(self) -> str:
        """Get package name from setup.cfg."""
        setup_cfg = self.root_dir / "setup.cfg"

        with open(setup_cfg) as f:
            for line in f:
                if line.startswith("name = "):
                    return line.split("=")[1].strip()
        raise ValueError("Package name not found in setup.cfg")

    def run_command(
        self, command: str, description: str = "", check: bool = True
    ) -> bool:
        """Run a shell command and handle errors."""
        print(f"\n📋 {description or command}")
        try:
            result = subprocess.run(
                command, shell=True, check=check, text=True, capture_output=True
            )
            if result.stdout:
                print(f"📤 Output:\n{result.stdout}")
            if result.returncode == 0:
                print(f"✅ Success: {description or command}")
                return True
            return False
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            print(f"📥 Error output:\n{e.stderr}")
            return False

    def check_environment(self) -> bool:
        """Check development environment."""
        commands = [
            ("python -V", "Python version"),
            ("pip list", "Installed packages"),
            (
                "tree -I 'venv|__pycache__|*.pyc|*.pyo|*.pyd|.git|"
                ".pytest_cache|*.egg-info|dist|build'",
                "Project structure",
            ),
        ]
        return all(self.run_command(cmd, desc) for cmd, desc in commands)

    def install_package(self) -> bool:
        """Install package in development mode."""
        commands = [
            (
                f"pip uninstall {self.package_name} -y",
                "Uninstalling old version",
                False,
            ),
            ("pip install -e .", "Installing package", True),
        ]
        return all(self.run_command(cmd, desc, chk) for cmd, desc, chk in commands)

    def run_tests(self) -> bool:
        """Run test suite."""

        # Ensure package is installed in development mode
        if not self.run_command(
            "pip show " + self.package_name,
            "Checking package installation",
            check=False,
        ):
            print("\n❌ Package not installed. Installing in development mode...")
            if not self.install_package():
                return False

        # Get the source directory for coverage
        src_dir = self.root_dir / "src" / self.package_name
        if not src_dir.exists():
            src_dir = self.root_dir / self.package_name

        # Run tests with coverage
        return self.run_command(
            f"pytest -v --cov={src_dir} --cov-report=term-missing tests/",
            "Running tests with coverage",
        )

    def format_code(self) -> bool:
        """Format code using black and isort."""
        commands = [
            ("black .", "Formatting with black"),
            ("isort .", "Sorting imports"),
        ]
        return all(self.run_command(cmd, desc) for cmd, desc in commands)

    def check_lint(self) -> bool:
        """Run linting checks."""
        commands = [("flake8 .", "Running flake8"), ("mypy .", "Running type checks")]
        return all(self.run_command(cmd, desc) for cmd, desc in commands)

    def show_changes(self) -> bool:
        """Show current git changes."""
        commands = [
            ("git status", "Current git status"),
            ("git diff", "Current changes"),
        ]
        return all(self.run_command(cmd, desc) for cmd, desc in commands)

    def clean_builds(self) -> bool:
        """Clean all build artifacts."""
        return self.run_command(
            "rm -rf dist build *.egg-info __pycache__ .pytest_cache .mypy_cache",
            "Cleaning build artifacts",
        )

    def build_package(self) -> bool:
        """Build package distributions."""
        if not self.clean_builds():
            return False

        return self.run_command("python -m build", "Building package")

    def upload_to_testpypi(self) -> bool:
        """Upload package to TestPyPI."""
        if not questionary.confirm(
            "Are you sure you want to upload to TestPyPI?"
        ).ask():
            return False

        print("\n🧪 Running tests before uploading to TestPyPI...")
        if not self.run_tests():
            print("❌ Tests failed! Aborting upload to TestPyPI.")
            return False

        print("\n🔨 Tests passed! Building package...")
        if not self.build_package():
            print("❌ Package build failed! Aborting upload to TestPyPI.")
            return False

        return self.run_command(
            "twine upload --repository testpypi dist/*", "Uploading to TestPyPI"
        )

    def upload_to_pypi(self) -> bool:
        """Upload package to PyPI."""
        if not questionary.confirm("Are you sure you want to upload to PyPI?").ask():
            return False

        print("\n🧪 Running tests before uploading to PyPI...")
        if not self.run_tests():
            print("❌ Tests failed! Aborting upload to PyPI.")
            return False

        print("\n🔨 Tests passed! Building package...")
        if not self.build_package():
            print("❌ Package build failed! Aborting upload to PyPI.")
            return False

        return self.run_command("twine upload dist/*", "Uploading to PyPI")

    def update_version(self) -> bool:
        """Update package version."""
        current_version = self._get_current_version()
        if not current_version:
            return False

        version_type = questionary.select(
            "Select version update type:",
            choices=["patch", "minor", "major"],
            style=self.style,
        ).ask()

        if not version_type:
            return False

        major, minor, patch = map(int, current_version.split("."))
        if version_type == "patch":
            patch += 1
        elif version_type == "minor":
            minor += 1
            patch = 0
        else:
            major += 1
            minor = 0
            patch = 0

        new_version = f"{major}.{minor}.{patch}"
        self._update_version_in_files(new_version)
        print(f"✅ Version updated: {current_version} → {new_version}")
        return True

    def _get_current_version(self) -> Optional[str]:
        """Get current package version."""
        setup_cfg = self.root_dir / "setup.cfg"
        if not setup_cfg.exists():
            print("❌ setup.cfg not found")
            return None

        with open(setup_cfg) as f:
            for line in f:
                if line.startswith("version = "):
                    return line.split("=")[1].strip()
        return None

    def _update_version_in_files(self, new_version: str) -> None:
        """Update version in configuration files."""
        module_name = self.package_name

        files_to_update = {
            "setup.cfg": (r"version = .*", f"version = {new_version}"),
            f"src/{module_name}/__init__.py": (
                r'__version__ = ".*"',
                f'__version__ = "{new_version}"',
            ),
        }

        import re

        for filename, (pattern, replacement) in files_to_update.items():
            filepath = self.root_dir / filename
            if not filepath.exists():
                continue

            with open(filepath) as f:
                content = f.read()

            content = re.sub(pattern, replacement, content)

            with open(filepath, "w") as f:
                f.write(content)

    def show_menu(self) -> None:
        """Display interactive menu."""
        while True:
            choices = []
            for group, commands in self.commands.items():
                if choices:
                    choices.append(questionary.Separator())
                choices.append(questionary.Separator(f"━━ {group} ━━"))
                choices.extend(list(commands.keys()))

            choices.extend([questionary.Separator("━" * 30), "Exit"])

            action = questionary.select(
                "Select action:",
                choices=choices,
                style=self.style,
                instruction="Use ↑↓ to navigate, Enter to select",
            ).ask()

            if not action or action == "Exit":
                break

            for commands in self.commands.values():
                if action in commands:
                    try:
                        commands[action]()
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        if not questionary.confirm("Continue?", style=self.style).ask():
                            return
                    break


if __name__ == "__main__":
    manager = DeployManager()
    manager.show_menu()
