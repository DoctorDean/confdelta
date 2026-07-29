# Releasing confdelta to PyPI

confdelta publishes to PyPI with **Trusted Publishing** (OpenID Connect): GitHub
Actions authenticates to PyPI directly, so there is **no API token stored in the
repository**. The workflow is [`.github/workflows/publish.yml`](../.github/workflows/publish.yml);
it runs when a GitHub Release is published, builds the sdist and wheel, checks
they render on PyPI, and uploads them.

Some steps below require the PyPI account owner and the GitHub repository admin —
they cannot be automated from a checkout because they involve logging in to PyPI
and creating credentials.

## One-time setup (claims the `confdelta` name on the first release)

1. **PyPI account.** Sign in (or register) at <https://pypi.org>.

2. **Register a "pending publisher"** so the very first upload is allowed to
   create the project and claim the name. On PyPI: *Account → Publishing → Add a
   new pending publisher*, and enter exactly:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `confdelta` |
   | Owner | `DoctorDean` |
   | Repository name | `confdelta` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

3. **Create the `pypi` environment** in GitHub: *Settings → Environments → New
   environment → `pypi`*. (You may add a required reviewer or a tag restriction;
   the environment name must match the workflow and the pending publisher.)

That is the whole token-free setup. It only has to be done once.

## Cutting a release

1. Make sure `master` is green and the version in
   [`src/confdelta/_version.py`](../src/confdelta/_version.py) is what you want
   (it is the single source of truth; `pyproject.toml` reads it dynamically).
   confdelta follows semantic versioning; `0.1.0` is the first public release.

2. Tag and publish a **GitHub Release** named for the version (e.g. tag
   `v0.1.0`, release title `v0.1.0`). Publishing the release triggers
   `publish.yml`, which builds and uploads to PyPI via OIDC. `publish.yml` must
   already be on the **default branch** for the release event to find it.

3. Watch the *Publish to PyPI* workflow in the Actions tab. On success the
   package appears at <https://pypi.org/project/confdelta/> and the name is
   secured.

4. Verify in a clean environment:

   ```bash
   python -m venv /tmp/confdelta-check && /tmp/confdelta-check/bin/pip install confdelta
   /tmp/confdelta-check/bin/python -c "import confdelta; print(confdelta.__version__)"
   ```

## Alternative: claim the name immediately with a manual upload

If you want the name reserved before wiring up Trusted Publishing, upload the
built distribution once with an API token. This uses a token (unlike the route
above), so do it yourself — do not paste tokens into shared tooling:

```bash
python -m build                 # writes dist/confdelta-0.1.0*
python -m twine check dist/*    # both files should PASS
python -m twine upload dist/*   # username: __token__   password: <your PyPI token>
```

After this, still complete the one-time Trusted Publishing setup so future
releases stay token-free.

## After the first release

- Connect the GitHub repo to **Zenodo** for a citable DOI, then add the DOI to
  [`CITATION.cff`](../CITATION.cff) and the README.
- Bump `src/confdelta/_version.py` for the next cycle.
