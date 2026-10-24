# Mercari Japan AI Shopper

Natural-language shopping agent for Mercari Japan. Requires Python 3.13+. Phase 1 adds the Mercari data layer (signed search, browser fallback, fixtures). The agent loop arrives in later phases.

```bash
python -m mercari_shopper --version
python -m mercari_shopper.sources --q "switch 有機EL" --max 5
python -m mercari_shopper.sources --q "switch 有機EL" --max 5 --source fixture
```
