# Slide plan

Viewing context: live class presentation on May 5, 2026.

Principle: the slides are prompts for the oral presentation, not a written report.  Each question starts with one prompt slide containing the assignment phrase, followed by short answer slides with one code idea or one plot each.

## Structure

1. Part 0: article
   - Explain the article's purpose.
   - One visual mechanism: MCMC draws -> score -> lower-variance estimate.
   - Visible message: the paper reduces numerical noise in posterior means.

2. Question 1: sampler and data
   - Prompt slide.
   - Code slide for GARCH simulation.
   - Code slide for random-walk Metropolis.
   - Plot slides for simulated and real chains.
   - Plot slide for repeated runs.

3. Question 2: first-order control variate
   - Prompt slide.
   - Code/formula slide for the gradient.
   - Plot slides for real and simulated variance comparisons.

4. Question 3: larger dictionaries and Lasso
   - Prompt slide.
   - Code slide for degree-2 controls.
   - Code slide for Lasso screening.
   - Plot slide for LSLASSO comparison.

5. Question 4: dependence in MCMC regression
   - Prompt slide.
   - Concept slide on non-iid regression rows.
   - Plot slide from the Q4 analysis on thinning/block means.

## Reveal.js implementation notes

- Local vendored Reveal.js files are in `vendor/reveal`.
- Local vendored KaTeX files are in `vendor/katex/dist`.
- Speaker notes use `<aside class="notes">`.
- Math is enabled through `RevealMath.KaTeX`.
- PDF export uses `?print-pdf`, then Chrome/Chromium print settings.

## Sources checked

- Reveal.js installation: https://revealjs.com/installation/
- Reveal.js markup: https://revealjs.com/markup/
- Reveal.js math plugin: https://revealjs.com/math/
- Reveal.js speaker view: https://revealjs.com/speaker-view/
- Reveal.js PDF export: https://revealjs.com/pdf-export/
