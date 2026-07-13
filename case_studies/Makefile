PYTHON ?= python
QUICK_OUTPUT_ROOT ?= outputs-quick
FULL_OUTPUT_ROOT ?= outputs-full
REFERENCE_ROOT ?= reference_results
.PHONY: quick full test verify verify-quick verify-reference clean

quick:
	$(PYTHON) run_studies.py --quick --output-root $(QUICK_OUTPUT_ROOT)

full:
	$(PYTHON) run_studies.py --full --output-root $(FULL_OUTPUT_ROOT)

test:
	$(PYTHON) -m pytest

verify: full test
	$(PYTHON) verify_repository.py --output-root $(REFERENCE_ROOT)
	$(PYTHON) verify_repository.py --output-root $(FULL_OUTPUT_ROOT) \
		--compare-reference $(REFERENCE_ROOT)

verify-quick: quick test
	$(PYTHON) verify_repository.py --output-root $(QUICK_OUTPUT_ROOT) \
		--allow-quick-nonclaim

# Verify the committed scientific payload without recomputing it. The worked
# examples have their own gate in the combined repository.
verify-reference:
	$(PYTHON) verify_repository.py --output-root $(REFERENCE_ROOT)

clean:
	$(PYTHON) scripts/clean_generated.py --output-root $(QUICK_OUTPUT_ROOT)
	$(PYTHON) scripts/clean_generated.py --output-root $(FULL_OUTPUT_ROOT)
