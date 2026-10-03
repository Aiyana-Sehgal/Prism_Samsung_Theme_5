"""Offline tests for the smart-home extension: correction + state-based idempotency."""
import asyncio, os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "agent")); sys.path.insert(0, os.path.join(here, "..", "extension"))
from controller import ToolController
from smart_home_agent import Home

def make():
    home = Home(); home.render = lambda *a, **k: None
    ctl = ToolController("t", lambda fn, **kw: getattr(home, fn)(**kw), lambda *a: None, quiet_window=0.3, dedupe=False)
    return home, ctl

async def test_correction_only_changes_bedroom():
    home, ctl = make()
    ctl.on_user_state("speaking"); ctl.on_user_state("listening")          # "set the AC to 22..." pause
    stale = asyncio.create_task(ctl.run("set_ac", {"room": "living room", "temperature": 22}))
    await asyncio.sleep(0.1); ctl.on_user_state("speaking"); ctl.on_user_state("listening")  # "no, bedroom, 24"
    await ctl.run("set_ac", {"room": "bedroom", "temperature": 24})
    assert (await stale)["status"] == "superseded"
    assert home.state["living room"]["ac"]["temperature"] == 25
    assert home.state["bedroom"]["ac"] == {"power": "on", "temperature": 24}
    assert len(home.actions) == 1

async def test_repeat_is_no_change_but_later_change_back_works():
    home, ctl = make()
    await ctl.run("set_ac", {"room": "living room", "temperature": 22})
    r = await ctl.run("set_ac", {"room": "living room", "temperature": 22})
    assert r["status"] == "no_change"
    await ctl.run("set_ac", {"room": "living room", "temperature": 24})
    r = await ctl.run("set_ac", {"room": "living room", "temperature": 22})   # legit repeat later
    assert r["status"] == "success" and len(home.actions) == 3

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): asyncio.run(f()); print("PASS", n)
