# Wesley wrote this
set -e
cd ~
[ -d nanochat ] || git clone -q https://github.com/karpathy/nanochat.git
cd nanochat
git log -1 --format='%h %cd %s' --date=short
ls
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
~/.local/bin/uv --version
