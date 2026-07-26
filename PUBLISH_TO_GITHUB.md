# Publish this repository to GitHub

## Browser method

1. Create a new empty GitHub repository, suggested name: `vlc-context-layer-reproducibility`.
2. Do not initialize it with a README, license, or `.gitignore` because those files are already included.
3. Copy the repository URL.
4. From the unpacked folder, run:

```bash
git remote add origin <REPOSITORY_URL>
git branch -M main
git push -u origin main --tags
```

## GitHub CLI method

After authenticating with `gh auth login`:

```bash
gh repo create vlc-context-layer-reproducibility --private --source=. --remote=origin --push
git push origin v1.0.0-r2
```

Change `--private` to `--public` only after the authors have approved the code, data, checkpoint, and licensing status.

## DOI after upload

To obtain a persistent DOI, connect the public GitHub repository to Zenodo, create a release, and replace all `REPLACE_WITH_PERSISTENT_REPOSITORY_URL` markers in:

- `README.md`
- `CITATION.cff`
- the manuscript Data and Code Availability statement;
- the response to Reviewer 1;
- the response to Reviewer 2.
