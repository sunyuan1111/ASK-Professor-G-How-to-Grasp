# Ask Professor G Project Site

This directory contains a static project website for **ASK Professor G: How to Grasp**. It is designed to work with GitHub Pages by setting the Pages source to the `docs/` folder.

## Files

- `index.html`: the project landing page.
- `styles.css`: all page styling.
- `assets/paper-fig*.png`: paper figures used by the project page.
- `assets/demo-video.mp4`: the demo video copied from the workspace root `vedio.mp4`.
- `media/`: generated demo images used directly by the project page. Keep these files in regular Git, not Git LFS, so GitHub Pages can serve them.

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
