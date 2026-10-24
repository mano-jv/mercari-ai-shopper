# Mercari Japan AI Shopper

Natural-language shopping agent for Mercari Japan. Requires Python 3.13+. Phase 4 adds the CLI, a top-3 presenter (plain / Rich / JSON), interactive mode, and a 10-query eval harness.

```bash
python -m mercari_shopper --version
python -m mercari_shopper --offline --plain -q "a used Nintendo Switch OLED under 30000 yen"
python -m mercari_shopper --offline --interactive
python -m mercari_shopper --eval
python -m mercari_shopper.sources --q "switch 有機EL" --max 5 --source fixture
```
