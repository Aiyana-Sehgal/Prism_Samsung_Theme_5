# Demo video script — 3 to 5 minutes

Single takes, no editing needed beyond trimming. Record screen + mic.
Target **4:00**. Timings below are cumulative.

**Before you start**
- Terminal 1: `wsl -d Ubuntu-22.04`, sized large, font bumped up so it reads on video
- Terminal 2: same, for the trace tail
- Deck open at slide 3 (architecture) in another window
- Close notifications

---

## 0:00 – 0:25 · The problem (talking head or slide 1)

> "Voice agents get interrupted. People say *'book me a flight to Paris on the tenth… no wait, make that Berlin.'*
>
> A normal agent has already called `search_flights` for Paris by the time you say Berlin. In Full-Duplex-Bench v3 that single stale call fails the scenario outright — even a read-only search counts as an extra tool call.
>
> My fix is one coordination layer: the language model only *proposes* tool calls. A controller decides whether each one actually runs."

## 0:25 – 0:50 · Architecture (slide 3)

> "Everything upstream is a stock LiveKit pipeline — Silero VAD, a semantic turn detector, Whisper, a local LLM.
>
> The difference is here. Every proposed call waits for a quiet window — 0.9 seconds of user silence measured from the end of their speech. If they speak again, the call is dropped before it executes and before it's logged. Identical calls run once. And the session stays alive after the harness disconnects, so calls still inside the gate can finish."

## 0:50 – 2:00 · Live correction (the core demo)

**Terminal 2**, start this first so it's already streaming:

```bash
tail -f /tmp/fdb_controller_trace.log
```

**Terminal 1**, from `~/fdb-controller/work/Full-Duplex-Bench/v3`:

```bash
python fdb_agent.py console
```

Wait for it to come up, then say, with a clear pause where marked:

> "Search for flights to Paris on the tenth …" **(pause ~1 second, let it start thinking)** "… no wait, make that Berlin."

Point at Terminal 2 as the lines appear:

> "There — `superseded`. The Paris call was proposed, it waited at the gate, I started speaking again, and it was dropped. It never executed and it never reached the tool log.
>
> And then `executed` for Berlin. One call, the right one. The benchmark only ever sees Berlin."

**If the correction doesn't trigger a `superseded` line**, say so honestly and retry once — the gate depends on the pause landing inside the quiet window. Don't fake it.

## 2:00 – 2:50 · Results (slide 5)

> "A hundred scenarios. Tool selection accuracy 0.895, against 0.803 for the published cascaded baseline. Interruption rate 4.3% against their 33%. Latency roughly half.
>
> The number I care about is this one. The FDB-v3 paper identifies self-correction as the cascaded architecture's worst failure mode — the published baseline scores 0.176, the lowest of any system they tested. This agent scores 0.471. Two point seven times better, using a free local 7-billion-parameter model instead of GPT-4o.
>
> And the same pattern shows up in the rollback split: 0.471 on scenarios with state rollback versus 0.253 without. It's *better on the harder subset*, which is the inverse of the usual pattern. That's the gate doing the work, not the model.
>
> One honest caveat: official evaluation uses a hosted GPT-4o judge, which costs money, so I didn't use it. 0.290 is the benchmark's stricter exact-match scoring; 0.500 is its judge pointed at my local model. The real number is somewhere between, and I've said so in the README."

## 2:50 – 3:40 · Extension (live)

```bash
cd ~/fdb-controller/extension && python smart_home_agent.py console
```

Three lines, pausing for the terminal to render home state each time:

1. > "Set the AC to 22 … **(pause)** no wait, the bedroom one, and make it 24."

   > "Only the bedroom changed. The living room call was superseded — same gate, different domain."

2. > "Set the bedroom AC to 24."

   > "`no_change`. The device is already in that state, so nothing happens."

3. > "Lock the front door."

   > "And a normal action still works."

Then the point that matters:

> "This is why the extension isn't just a reskin. In the benchmark, tools are stateless, so deduplicating per session is correct. In a real home that's actively wrong — 'turn the lights back on' an hour later has to work. So idempotency moves into the device: already in that state returns `no_change`, but a genuine repeat later still runs. The controller takes that as a parameter."

## 3:40 – 4:00 · Limitation and close

> "The honest limitation: argument accuracy, 0.418. Tool selection is nearly solved at 0.895 — the agent picks the right tool and then writes the date in the wrong format. `2023-10-07` where the user said 'October 7'. That one class of error zeroes out an entire domain.
>
> About half of those failures are formatting the model can be prompted out of; the rest want a bigger model. Tool selection being solved while arguments aren't tells me exactly where the next effort goes.
>
> Everything reproduces with one command, and needs no paid API key."

---

## Fallback if the microphone won't work

Record the **benchmark** portion instead of live speech:

```bash
tail -f /tmp/fdb_controller_trace.log    # terminal 2
./quick_eval.sh demo                      # terminal 1
```

Narrate the `superseded` / `executed` / `dedup` decisions as they stream past, then show
`results/controller_pass_rate_report.json`. Say plainly that it's the recorded benchmark audio rather
than a live microphone — that's a normal thing to show and far better than a staged take.

## Checklist

- [ ] 3–5 minutes
- [ ] Shows a correction being superseded, live, in the trace
- [ ] Shows the extension: correction, `no_change`, normal action
- [ ] States the self-correction result against the published baseline
- [ ] States the judge caveat out loud
- [ ] States one real limitation
