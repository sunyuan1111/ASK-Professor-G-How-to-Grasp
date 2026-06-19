# Ask Professor G Project Site

This directory contains a static project website for **ASK Professor G: How to Grasp**. It is designed to work with GitHub Pages by setting the Pages source to the `docs/` folder.

## Files

- `index.html`: the project landing page.
- `styles.css`: all page styling.
- `assets/ask-professor-g-paper.pdf`: the paper PDF copied from `root.pdf`.
- `assets/paper-fig*.png`: figures extracted from the paper PDF.
- `assets/demo-video.mp4`: the demo video copied from the workspace root `vedio.mp4`.

## Preview Locally

From the repository root:

```bash
python -m http.server 8000 -d docs
```

Then open:

```text
http://localhost:8000
```

The page is intentionally dependency-free, so it can also be opened directly from `docs/index.html` for a quick check.
