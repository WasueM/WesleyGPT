# Wesley wrote this
cd ~/nanochat && source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
for q in "Hi! Who are you?" "Why is the sky blue?" "What is the capital of France?" "Write a short poem about a cat." "What is 12 + 7?"; do
  echo "######## USER: $q"
  python -m scripts.chat_cli -i sft -g d12 -p "$q" 2>/dev/null | grep -vE "^Autodetected|^Loading|^Building|^No model|^Model|^Using|^\s*$" | tail -12
done
