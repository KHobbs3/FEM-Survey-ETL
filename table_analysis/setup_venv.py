import subprocess
import sys
from pathlib import Path

def run(cmd, quiet=True):
    """Run a command with optional quiet mode."""
    kwargs = dict(check=True)
    if quiet:
        kwargs.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(cmd, **kwargs)

venv_path = Path('.venv')
if not venv_path.exists():
    print('⚙️  Creating virtual environment...')
    subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)

python_exec = venv_path / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')

req_file = Path('requirements.txt')
if req_file.exists():
    print('📦 Syncing dependencies...')
    try:
        run([
            str(python_exec),
            '-m', 'pip', 'install',
            '-r', 'requirements.txt'
        ], quiet=True)
    except subprocess.CalledProcessError:
        print('❌ Dependency installation failed. Check the above error log.')
        sys.exit(1)
else:
    print('ℹ️  No requirements.txt found. Skipping install.')

print('✅ Environment ready.')
