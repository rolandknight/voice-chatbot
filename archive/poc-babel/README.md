# poc-babel — af_heart reference-clip candidates

The main app's `babel` persona is Kokoro `af_heart` (`../personas.yaml`).
Kokoro voices are embeddings, not recordings, so there is no sample to hand
to Chatterbox. This renders three ~10 s `af_heart` clips (CPU only) so one
can be picked as the Chatterbox reference for a cloned `babel` voice.

    make          # setup (downloads ~340 MB of Kokoro model files) + render
    make test     # unit tests for the trim/normalize helpers
    ls out/       # babel-{a-intro,b-narration,c-dialogue}.{mp3,wav} + manifest.json

Listen to the mp3s; the matching `.wav` is the lossless one to copy into
`../voices/` for Chatterbox. Texts, speeds and rationale live in
`render_variants.py:VARIANTS`.
