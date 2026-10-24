# Mercari Japan AI Shopper

Natural-language shopping agent for Mercari Japan. Requires Python 3.13+. Phase 3 adds the tool-calling loop: grounded top-3, one repair on hallucinated ids, and step/tool/time budgets.

```bash
python -m mercari_shopper --version
python -m mercari_shopper.sources --q "switch 有機EL" --max 5
python -m mercari_shopper.sources --q "switch 有機EL" --max 5 --source fixture
python -m mercari_shopper.agent --q "a used Nintendo Switch OLED under 30000 yen" --offline
```
