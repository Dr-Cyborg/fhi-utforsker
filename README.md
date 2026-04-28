# FHI-utforsker

Interaktiv Streamlit-app for å utforske helsedata fra Folkehelseinstituttet (FHI).

Henter data live fra [FHI Statistikk Open API](https://statistikk-data.fhi.no/api/open/v1)
— legemiddelregisteret, dødsårsaksregisteret, MSIS, MFR, abortregisteret, m.fl.

## Funksjoner

- Bla i alle FHI-kilder og tabeller
- Hierarkisk tre for forgrenede dimensjoner (ATC, ICD-koder)
- Fra-til-velger for år/perioder
- Velg fritt hva som er x-akse, farge og form
- Linje, søyle, areal, scatter
- Oversikt-fane som auto-genererer småmultipler for alle dimensjoner

## Kjør lokalt

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Klar for [Streamlit Community Cloud](https://share.streamlit.io) — pek på `app.py`.
