# Slide plan

Viewing context: live class presentation on May 5, 2026.

Principle: the slides are prompts for the oral presentation, not a written report.  Each question gets one slide only.  Speaker notes can hold reminders, but visible text stays short.

## Structure

1. Part 0: article
   - Explain the article's purpose.
   - One visual mechanism: MCMC draws -> score -> lower-variance estimate.
   - Visible message: the paper reduces numerical noise in posterior means.

2. Question 1: sampler and data
   - One slide.
   - Final content should come from the final notebook.
   - Best likely visual: trace/running mean or data overview.

3. Question 2: first-order control variate
   - One slide.
   - Show the regression idea and one variance comparison.
   - Avoid too many equations on the slide.

4. Question 3: larger dictionaries and Lasso
   - One slide.
   - Show why OLS gets expensive/unstable and how Lasso screens controls.
   - Best likely visual: variance-ratio comparison or dictionary-growth plot.

5. Question 4: dependence in MCMC regression
   - One slide.
   - Use the Q4 analysis only if it remains consistent with the final notebook.
   - Best likely visual: split/thin/block comparison or autocorrelation reduction.

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
