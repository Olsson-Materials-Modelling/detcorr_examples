.PHONY: run test test-integration

PYTHON ?= python
OUTPUTS ?= $(abspath ../../regenerated/fulton-demonstrator)
MPLCONFIGDIR ?= $(or $(TMPDIR),/tmp)/detcorr-matplotlib-cache
EXAMPLE_PYTHONPATH := $(CURDIR)/src:$(CURDIR)/../detcorr_demo_common/src:$${PYTHONPATH:-}

run:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" PYTHONPATH="$(EXAMPLE_PYTHONPATH)" $(PYTHON) src/fulton_market_detcorr_demo.py --outputs "$(OUTPUTS)"

test:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" PYTHONPATH="$(EXAMPLE_PYTHONPATH)" $(PYTHON) -m pytest -q tests

test-integration:
	DETCORR_EXAMPLES_REQUIRE_DETCORR=1 MPLCONFIGDIR="$(MPLCONFIGDIR)" PYTHONPATH="$(EXAMPLE_PYTHONPATH)" $(PYTHON) -m pytest -q tests
