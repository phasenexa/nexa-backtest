"""Deliberately broken syntax — should fail step 1 and skip remaining steps."""
from nexa_backtest.algo import SimpleAlgo

class BrokenAlgo(SimpleAlgo
    def on_setup(self, ctx):
        pass
