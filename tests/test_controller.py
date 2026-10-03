"""Offline tests for ToolController: no LiveKit, no API keys needed."""
import asyncio, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
from controller import ToolController

def make():
    logged = []
    ctl = ToolController(room="t", execute_fn=lambda fn, **kw: {"status": "success", "fn": fn, **kw},
                         log_fn=lambda fn, a, t0, t1: logged.append((fn, a)), quiet_window=0.3)
    return ctl, logged

async def test_self_correction_drops_stale_call():
    ctl, logged = make()
    ctl.on_user_state("speaking"); ctl.on_user_state("listening")      # "flights to Paris —" then pause
    stale = asyncio.create_task(ctl.run("search_flights", {"destination": "Paris", "date": "Sep 10"}))
    await asyncio.sleep(0.1)
    ctl.on_user_state("speaking"); await asyncio.sleep(0.1); ctl.on_user_state("listening")  # "...make that Berlin"
    fresh = await ctl.run("search_flights", {"destination": "Berlin", "date": "Sep 10"})
    assert (await stale)["status"] == "superseded"
    assert fresh["status"] == "success"
    assert logged == [("search_flights", {"destination": "Berlin", "date": "Sep 10"})], logged

async def test_duplicate_state_change_runs_once():
    ctl, logged = make()
    a = {"product_id": "PROD1", "quantity": 3}
    r1, r2 = await asyncio.gather(ctl.run("add_to_cart", a), ctl.run("add_to_cart", dict(a)))
    r3 = await ctl.run("add_to_cart", {"product_id": "prod-1", "quantity": 3.0})   # normalized duplicate
    assert len(logged) == 1, logged
    assert "Already done" in r3["note"]

async def test_same_tool_different_args_both_run():
    ctl, logged = make()
    await ctl.run("track_order", {"order_id": "CAT"}); await ctl.run("track_order", {"order_id": "DOG"})
    assert len(logged) == 2

async def test_interrupted_speech_handle_blocks_call():
    ctl, logged = make()
    r = await ctl.run("book_flight", {"passenger_name": "Robin"}, is_interrupted=lambda: True)
    assert r["status"] == "superseded" and not logged

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            asyncio.run(fn()); print("PASS", name)
