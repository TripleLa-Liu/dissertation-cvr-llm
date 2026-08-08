# Small patches needed in Chapters 1, 2, and 5

These are quoted verbatim from the compiled PDF (I don't have editable `.tex`
source for these chapters, only `methods_results_draft.tex` / the new
`methods_draft.tex` / `results_draft.tex`), so find-and-replace the exact text
below in the live Overleaf source. Reason: Ch1/2/5 currently describe the
Dynamic Graph layer as unimplemented future work, which is no longer true now
that V4 exists (Section 3.3 / 4.2.3 in the restructured version). Delayed
Feedback Correction is still accurately described as out of scope — no change
needed there beyond one optional addition noted below.

---

## 1. Chapter 1, Introduction (end of chapter, p.12)

**Current text:**

> RQ2 and RQ3 are addressed only partially, through the cold-start segmentation
> analysis in Section 4.2 and the discussion of encoder cost in Section 4.1; a
> full delayed-feedback correction layer and a dynamic graph layer over the
> interaction sequence remain future work, returned to in Chapter 5.

**Suggested replacement:**

> RQ2 and RQ3 are addressed only partially, through the cold-start segmentation
> analysis and the discussion of encoder cost in Chapter 4. A scoped-down
> dynamic graph layer (a single-channel target-attention aggregator over each
> user's interaction history, evaluated on Amazon Reviews'23 only, since
> Ali-CCP carries no interaction timestamps) is implemented and evaluated in
> Section 4.2.3, though — as reported there — it does not yield a significant
> improvement on this dataset. A full delayed-feedback correction layer remains
> future work, returned to in Chapter 5.

Note: the "Section 4.2 / Section 4.1" cross-references need updating regardless
of this specific sentence, since the whole chapter has been renumbered under
the new dataset-first structure (see `results_draft.tex`) — the fair-copy
version of this paragraph should point at whatever the final section labels
end up being (`\ref{sec:results-v3-aliccp}`, `\ref{sec:results-encoders}` /
`\ref{sec:results-v4}` in the new draft).

---

## 2. Chapter 2, Section 2.2 Delayed Feedback Correction (p.14) — optional

**Current text (still accurate, no change required):**

> The dissertation does not implement a delayed-feedback correction layer of
> its own – Ali-CCP, the primary dataset used here, does not carry the
> timestamp information a delayed-feedback correction method would need – but
> this literature remains directly relevant to the third layer of the
> architecture sketched in Chapter 5, and to RQ2's question of whether
> LLM-derived user representations might independently reduce the same
> false-negative noise these methods are built to correct.

**Optional addition** (only if you want to preempt the obvious follow-up
question — Amazon *does* have timestamps now, so why not delayed feedback
there too):

> ...this literature remains directly relevant to the third layer of the
> architecture sketched in Chapter 5, and to RQ2's question of whether
> LLM-derived user representations might independently reduce the same
> false-negative noise these methods are built to correct. Amazon Reviews'23
> (Section 3.1.2), used for the dynamic graph experiments in Section 4.2.3,
> does carry genuine timestamps, but delayed-feedback correction specifically
> requires modelling a *conversion-lag* distribution that this dataset's
> single-signal (verified-purchase) label structure does not straightforwardly
> provide, so this component remains out of scope on both datasets.

---

## 3. Chapter 2, Section 2.3 Graph Neural Networks for Recommendation (p.15)

**Current text:**

> This body of work corresponds to the second layer of the architecture
> originally proposed for this dissertation – a dynamic graph network sitting
> on top of whatever user and item representations the LLM encoder layer
> produces – and DGSR in particular was the intended starting point, given its
> direct precedent for combining sequence and graph structure. As with the
> delayed-feedback layer, building and evaluating this component was out of
> scope for the work completed so far, which instead isolates the encoder
> question in Chapter 3; a graph layer of this kind is the natural next step
> once an encoding strategy has been selected, discussed further in Chapter 5.

**Suggested replacement:**

> This body of work corresponds to the second layer of the architecture
> originally proposed for this dissertation – a dynamic graph network sitting
> on top of whatever user and item representations the LLM encoder layer
> produces – and DGSR (Zhang et al., 2022) in particular was the intended
> starting point, given its direct precedent for combining sequence and graph
> structure. Section 3.2 (Models) implements a deliberately scoped-down version
> of this layer: a single-channel target-attention aggregator over a user's causally-
> ordered interaction history, in the spirit of DIN's (Zhou et al., 2018)
> target-attention rather than DGSR's full dual long/short-term channels and
> edge-quintuple formulation, combined with the LLM-alignment pattern from
> Chapter 3. It is evaluated only on Amazon Reviews'23, since Ali-CCP carries
> no interaction timestamps and therefore admits no causally-ordered history.
> Section 4.2.3 reports the result: the mechanism trains stably and learns a
> non-trivial recency-decay rate, but produces no significant improvement over
> the ID or LLM-aligned baselines on this dataset, most likely because per-user
> history here is too sparse (a median of 1–2 available items) for a
> sequence-aggregation mechanism to have much to work with — a negative result
> discussed further in Chapter 5.

---

## 4. Chapter 5, Discussion, closing sentence (p.26)

**Current text:**

> This reframes the dissertation's original three-pattern comparison (REPLACE
> vs. ALIGN vs. ID-only) as a supervision-source question rather than a pure
> architecture question, and is the most direct empirical link so far between
> this dissertation's LLM-encoder experiments and its broader thesis that
> different information sources are useful in different, identifiable regimes
> – a link the planned Dynamic Graph and Delayed Feedback Correction
> components are intended to extend further.

**Suggested replacement:**

> This reframes the dissertation's original three-pattern comparison (REPLACE
> vs. ALIGN vs. ID-only) as a supervision-source question rather than a pure
> architecture question, and is the most direct empirical link so far between
> this dissertation's LLM-encoder experiments and its broader thesis that
> different information sources are useful in different, identifiable regimes.
> The Dynamic Graph layer (Section 3.2, results in Section 4.2.3) tests this
> thesis further and complicates it: on Amazon Reviews'23, a temporal
> attention mechanism over user history produces no significant gain on top of
> either the ID baseline or the LLM-aligned model, most plausibly because this
> dataset's interaction histories are too sparse for a sequence-aggregation
> mechanism to exploit. Where V3's routing result shows *which* information
> source helps depends on identifiable per-row regime, the Dynamic Graph
> result shows this is not unconditionally true of every additional
> information source — a graph/sequence signal only helps where the data
> density actually supports it, a boundary condition worth stating explicitly
> rather than assuming every architecturally-motivated addition pays off.
> Delayed Feedback Correction remains future work, for the reasons given in
> Section 2.2.

---

## Also worth doing while touching these chapters (not requested yet, flagging only)

Still empty/placeholder in the current PDF: Chapter 6 Conclusion ("Text."),
Appendix A Additional Results ("Text."), Abstract, Impact Statement, Thesis
Declaration, Acknowledgements. Not touched here — say the word if you want a
first pass at any of these next.
