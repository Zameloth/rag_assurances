# Public French insurance corpora — survey

> Research note for [issue #3](https://github.com/Zameloth/rag_assurances/issues/3):
> *What public French insurance corpora exist, and what shape are they in?*
>
> **Date of investigation:** 2026-08-01. All figures below marked *(measured)* were obtained by
> actually downloading and parsing the source on that date, not read off a blog post.
> Licence quotes are verbatim from the publisher's own pages.
>
> **Constraint driving this survey:** the corpus must be usable in a **public** GitHub portfolio
> repo. So for every source the first question is *may this be redistributed*, not *can I get it*.

---

## TL;DR table

| Source | Licence — can it go in a public repo? | Access | Format | Volume (measured) | Parse | FR quality |
|---|---|---|---|---|---|---|
| **service-public.fr fiches** | ✅ Licence Ouverte 2.0 | Single ZIP, no auth | XML | 5 557 files / 141 MB; **79 insurance docs ≈ 102 k words** | Easy | Excellent |
| **Code des assurances (LEGI/API)** | ✅ Licence Ouverte 2.0 | PISTE API (OAuth) *or* bulk tarball | JSON / XML | **2 377 in-force articles**; 8 692 article-versions | Medium | Excellent (legal register) |
| **Code des assurances (HF mirrors)** | ✅ CC-BY-4.0 / Apache-2.0 label | `datasets` / parquet | Parquet | 2 377 rows / 3.1 MB | Trivial | Excellent |
| **EUR-Lex (Solvabilité II, DDA)** | ✅ CC-BY-4.0 | HTTP / SPARQL | HTML/XML/PDF | ~2 directives | Easy | Excellent |
| **ACPR registre officiel** | ⚠️ No licence published — link, don't redistribute | Scrape w/ browser UA | PDF | 17 recommandations + more | Easy | Excellent |
| **Insurer CG (AXA/MAIF/Macif)** | ❌ Copyrighted; MAIF explicitly forbids | Direct PDF URLs | PDF | **6 docs ≈ 239 k words** | Medium | Excellent (real product language) |
| **France Assureurs** | ❌ Requires express prior agreement | Web | PDF | Statistics/reports | — | — |
| **data.gouv.fr "assurance"** | mixed | API | CSV/XLSX | 15 datasets, none textual | — | — |

---

## 1. Légifrance — Code des assurances

### 1.1 Is there a usable public API? Yes, via PISTE.

Primary source: the official [Légifrance API FAQ](https://www.legifrance.gouv.fr/contenu/pied-de-page/foire-aux-questions-api).

* **Auth:** OAuth 2.0 *client credentials* (RFC 6749 §4.4). You register free at
  [piste.gouv.fr](https://piste.gouv.fr/registration), accept the Légifrance API CGU
  (*API > Consent CGU API*), create an application, and get a `client_id` / `client_secret`.
* **Endpoints (production):**
  * token — `https://oauth.piste.gouv.fr/api/oauth/token`
  * base — `https://api.piste.gouv.fr/dila/legifrance/lf-engine-app`
  * sandbox equivalents live under `sandbox-oauth.piste.gouv.fr` / `sandbox-api.piste.gouv.fr`.
    Sandbox tokens are **not** valid in production, and sandbox serves *test* data only.
* **Token TTL:** `expires_in: 3600` seconds.
* **Friction to be aware of:** sandbox applications are created automatically; **production
  applications are created manually** (i.e. reviewed). Budget for a wait before you get real data.
* **Relevant methods:** `POST /search`, `POST /consult/getArticle`, `POST /consult/legiPart`,
  `GET /list/ping`. Fonds include `CODE_ETAT` / `CODE_DATE` (codes by status or by version date).
* **Granularity:** article-level. Articles are addressed by `LEGIARTI…` ids; the Code des assurances
  itself is `LEGITEXT000006073984`.
* **Response format:** JSON.

**Rate limits — flagged uncertainty.** No published number was found. The PISTE general CGU
(§3.1.3) only says:

> « L'AIFE établit et applique des limites d'utilisation des API (par exemple en limitant le nombre
> de demandes d'API qui peuvent être effectuées ou le nombre d'utilisateurs qui peuvent être servis),
> à la discrétion du Fournisseur. »
> — [PISTE CGU](https://piste.gouv.fr/en/?format=raw&option=com_apiportal&view=terms)

The Légifrance FAQ adds only that production applications "benefit from higher quotas". Concrete
per-second/per-day figures appear to be documented per-API inside the PISTE portal after login.
**Do not design an ingestion pipeline that assumes unlimited API throughput.**

### 1.2 Licensing — the clearest "yes" in this whole survey

The legal basis is the
[Arrêté du 24 juin 2014](https://www.legifrance.gouv.fr/loda/id/JORFTEXT000029135221), whose
article 1 lists the freed databases — « JORF », « LEGI », « KALI », « JADE », « CONSTIT »,
« CASS-INCA », « CAPP », « CNIL » et « CIRCULAIRES » — and whose article 2 states:

> « La réutilisation des données est soumise au respect d'une licence gratuite. »

That licence is **Licence Ouverte / Open Licence v2.0 (Etalab)**. The DILA's own dataset sheet for
LEGI ([`DILA_LEGI_Presentation_20170824.pdf`](https://echanges.dila.gouv.fr/OPENDATA/LEGI/DILA_LEGI_Presentation_20170824.pdf))
spells out both the licence and the attribution obligations:

> « Les données sont réutilisables gratuitement sous licence ouverte v2.0.
> Les réutilisateurs s'obligent à mentionner :
> – la paternité des données (DILA) ;
> – l'URL d'accès longue de téléchargement ;
> – le nom du fichier téléchargé ainsi que la date du fichier. »

➡️ **Redistribution in a public repo is explicitly permitted**, provided that three-part attribution
is carried (DILA + long download URL + filename & date). Record those three fields at ingest time —
they are a licence condition, not a nicety.

### 1.3 Bulk route (no API key needed)

Everything is on an open Apache directory index: <https://echanges.dila.gouv.fr/OPENDATA/LEGI/>

*(measured, 2026-08-01)*

* Full stock: `Freemium_legi_global_20250713-140000.tar.gz` — **1.1 GB**, **5 253 903 tar members**.
* **386** daily incremental tarballs currently retained (`LEGI_YYYYMMDD-HHMMSS.tar.gz`, 0.6–12 MB each).
* Update cadence per the DILA sheet: daily; texts consolidated "au plus tard dans les 3 jours ouvrés
  après publication au JORF". A fresh full stock is rebuilt "au minimum une fois par an", and
  first-time integrators are obliged to start from the latest stock.
* **Code des assurances share of the stock (measured by streaming the 1.1 GB tarball):**
  12 595 members under `LEGITEXT000006073984`, of which **8 692 article XML files, ≈ 57 MB
  uncompressed**. That count includes every historical version (`VIGUEUR`, `MODIFIE`, `ABROGE`),
  which is why it dwarfs the ~2 400 currently-in-force articles.

**Shape of the XML** — one file per *article version*, e.g.
`legi/global/code_et_TNC_en_vigueur/code_en_vigueur/LEGI/TEXT/00/00/06/07/39/LEGITEXT000006073984/article/LEGI/ARTI/…/LEGIARTI000006791829.xml`:

```xml
<ARTICLE>
 <META><META_COMMUN><ID>LEGIARTI…</ID><NATURE>Article</NATURE></META_COMMUN>
  <META_SPEC><META_ARTICLE>
     <NUM>L113-3</NUM><ETAT>VIGUEUR</ETAT>
     <DATE_DEBUT>…</DATE_DEBUT><DATE_FIN>2999-01-01</DATE_FIN>
  </META_ARTICLE></META_SPEC></META>
 <CONTEXTE><TEXTE …><TITRE_TXT …/>
    <TM><TITRE_TM id="LEGISCTA…">Livre Ier : Le contrat</TITRE_TM>
       <TM><TITRE_TM …>Titre Ier : Règles communes…</TITRE_TM></TM></TM>
 </TEXTE></CONTEXTE>
 <VERSIONS>…</VERSIONS><LIENS>…</LIENS>
</ARTICLE>
```

This is **very good RAG metadata**: article number, in-force state, validity window, and a full
hierarchical breadcrumb, all machine-readable, plus outbound legal links.

**Parse difficulty: medium.** The XML itself is trivial; the cost is operational — untar 1.1 GB /
5.2 M small files, filter to one `LEGITEXT`, then resolve "which version is in force at date *D*"
from `DATE_DEBUT`/`DATE_FIN`/`ETAT`.

### 1.4 Scraping legifrance.gouv.fr — don't

`robots.txt` only disallows `/download/`, but the site is behind bot protection: `curl` with a
realistic desktop User-Agent still returns **HTTP 403** on `/codes/texte_lc/LEGITEXT000006073984/`
*(measured)*. Use the API or the bulk dumps.

### 1.5 Ready-made mirrors (fastest path to a working corpus)

Both are third-party redistributions of Licence-Ouverte data — legitimate, but the *licence label
on the dataset card is the uploader's declaration*, not DILA's. Attribute DILA regardless.

**`louisbrulenaudet/code-assurances`** — [HF](https://huggingface.co/datasets/louisbrulenaudet/code-assurances) *(measured via the HF datasets-server)*

* **2 377 rows**, 3.1 MB parquet, single `train` split, DOI `10.57967/hf/1455`, updated 2025-09-21.
* Every row is `etat = VIGUEUR` (2 375 reported by the filter API; the remainder are edge states).
  Breakdown by article prefix: **L 937 · R 1 142 · A 246 · D 52** — i.e. the *complete in-force*
  Code des assurances, legislative + regulatory (décrets *and* arrêtés).
* Fields include `ref` ("Code des assurances, art. L100-1"), `num`, `texte`, `texteHtml`, `etat`,
  `dateDebut`, `cid`, and — most usefully — `fullSectionsTitre`
  (`"Partie législative > Livre Ier : Le contrat > Titre Ier : … > Chapitre Ier : …"`).
  The field names are a 1:1 match with the Légifrance API's article payload, confirming provenance.
* Card licence: `apache-2.0`. **Note the discrepancy** — the underlying content is Licence Ouverte
  2.0 from DILA. Apache-2.0 is not wrong in effect (both permit redistribution with attribution) but
  it is the uploader's re-declaration. Cite DILA + the arrêté, not just the HF card.

**`harvard-lil/cold-french-law`** — [HF](https://huggingface.co/datasets/harvard-lil/cold-french-law) *(measured)*

* **841 761 rows**, 985 MB parquet (2.34 GB original CSV), licence **CC-BY-4.0**, from the
  Harvard Library Innovation Lab. Covers *all* French codes, including the Code des assurances.
* Columns: `article_identifier`, `article_num`, `article_etat`, `article_date_debut`,
  `article_date_fin`, `texte_nature`, `texte_titre`, `texte_contexte` (newline-separated breadcrumb),
  `article_contenu_markdown`, `article_contenu_text`, plus `*_en` English translations.
* Filter on `texte_titre == "Code des assurances"` to get the same material as above with a cleaner
  licence story and Markdown-ready content. Cost: you download ~1 GB to keep ~3 MB.
* *(The HF filter endpoint returned HTTP 500 for the string filters I tried, so the exact
  Code-des-assurances row count in this dataset is **unverified**; do the filter locally.)*

---

## 2. service-public.fr — the best-value source in this survey

### 2.1 Bulk access, no scraping needed

The full consumer-facing knowledge base is published as **one ZIP**, refreshed daily:

* Dataset: [Fiches pratiques et ressources de Service-Public.gouv.fr Particuliers](https://www.data.gouv.fr/datasets/fiches-pratiques-et-ressources-de-service-public-gouv-fr-particuliers)
  (publisher: *Premier ministre* / DILA).
* Resource: <https://lecomarquage.service-public.gouv.fr/vdd/3.5/part/zip/vosdroits-latest.zip>
* Schemas (XSD): <https://echanges.dila.gouv.fr/OPENDATA/SERVICE-PUBLIC_DTD/schema_3.5.zip>
* A parallel business-facing dataset exists —
  [Fiches pratiques et ressources Entreprendre](https://www.data.gouv.fr/datasets/fiches-pratiques-et-ressources-entreprendre-service-public-gouv-fr) — same licence.

*(measured, 2026-08-01)*

* ZIP **23.5 MB** → **5 557 XML files**, **141 MB uncompressed**.
* Composition: **3 000** `F*` fiches pratiques, **2 305** `R*` ressources (forms, online procedures,
  letter templates), **243** `N*` theme/dossier nodes, plus index files
  (`arborescence.xml`, `menu.xml`, `questionsReponses.xml`, `redirections.xml`…).

### 2.2 How much of it is about insurance?

*(measured)*

* **163** documents have "assur" in `<dc:title>`.
* **79** documents sit under the *Assurance* / *Épargne* sub-theme or under an
  *Assurance habitation / automobile / vie* dossier — **≈ 101 700 words**. This is the tight,
  high-precision insurance subset.
* 7 dossier nodes: `N32` Assurance automobile, `N44` Assurance habitation, `N89` Assurance vie,
  plus `N31331` (associations), `N31348` / `N423` / `N31750` (assurance maladie).
* A broader "mentions *assur* anywhere" filter matches 1 379 `F*` fiches / 3.65 M words, but that is
  polluted by *assurance maladie*, *assuré social*, *s'assurer que* — not a usable boundary.

Sample titles: *Modification du contrat d'assurance habitation*, *Bonus-malus dans l'assurance
automobile*, *Assurance auto : qu'est-ce que la garantie responsabilité civile ?*,
*Catastrophe naturelle ou technologique : indemnisation par l'assurance*,
*Quel est le délai de prescription en matière d'assurance habitation ?*

### 2.3 Shape — unusually RAG-friendly

```xml
<Publication ID="F2594" type="Fiche d'information conditionnée"
             spUrl="https://www.service-public.gouv.fr/particuliers/vosdroits/F2594">
  <dc:title>Modification du contrat d'assurance habitation</dc:title>
  <dc:description>Les modifications du contrat d'assurance peuvent être demandées…</dc:description>
  <dc:date>modified 2025-04-28</dc:date>
  <dc:source>https://www.legifrance.gouv.fr/codes/id/LEGISCTA000006157200, …</dc:source>
  <dc:rights>https://www.service-public.gouv.fr/a-propos/mentions-legales</dc:rights>
  <FilDAriane>… <Niveau ID="N44">Assurance habitation</Niveau> …</FilDAriane>
  <SousThemePere ID="N20263">Assurance</SousThemePere>
  <DossierPere ID="N44"><SousDossier ID="N44-DA"><Fiche ID="F2591">Souscription</Fiche>…</SousDossier></DossierPere>
  <Introduction><Texte><Paragraphe>…</Paragraphe></Texte></Introduction>
  <Chapitre><Titre>…</Titre><BlocCas><Cas>…<Situation>…</Situation></Cas></BlocCas></Chapitre>
</Publication>
```

Why this matters:

* `<Chapitre>` / `<Titre>` / `<Paragraphe>` give **natural semantic chunk boundaries** — no
  heuristic splitter needed.
* `<dc:source>` **cross-links each fiche to the Légifrance sections it is based on** — free
  consumer-language ↔ legal-text alignment, which is exactly the join a two-register insurance RAG
  wants.
* `<FilDAriane>` / `<DossierPere>` give ready-made topical metadata for filtering.
* `<Cas>` / `<Situation>` encode conditional branches ("si vous êtes locataire / propriétaire"),
  which are the things a naive chunker usually destroys.

**Parse difficulty: easy.** Plain XML, UTF-8, no namespaces beyond Dublin Core, stable IDs.

### 2.4 Licensing

data.gouv.fr records the dataset licence as `fr-lo` = **Licence Ouverte v2.0**. The dataset page's
own reuse note requires acknowledging *"Service-Public.gouv.fr / DILA"* plus the complete download
URL, the filename and the file date — same three-part attribution as LEGI.
service-public.gouv.fr's own legal page states:
*« Sauf mention contraire, tous les contenus de ce site sont sous licence etalab-2.0. »*

➡️ **Safe to redistribute in a public repo**, with attribution.

### 2.5 Operational caveats

* **Flux versions expire.** Only two versions are kept in parallel; the older is decommissioned after
  6 months. v3.3 was removed on 2026-05-04; **v3.5 is current**, v3.4 still maintained. Pin the
  version in the URL and expect to bump it roughly annually.
* DILA commits to backward compatibility *within* a version (fields never removed or re-meant) but
  **new fields may appear at any time** — parse permissively.
* `robots.txt` on service-public.fr does not block crawling, but there is no reason to scrape: the
  dump is complete and daily.

### 2.6 French-language quality

Excellent. Professional editorial French, written for a lay audience, consistent house style,
dated and maintained. This is the single best source of *consumer-register* insurance French
available under an open licence.

---

## 3. Insurer conditions générales (CG)

### 3.1 Are they reliably published as public PDFs? Mostly yes — for mutuals

All of the following were downloaded anonymously, HTTP 200, no login, no paywall *(measured, 2026-08-01)*:

| Insurer | Document | URL | Pages | Words | Text layer |
|---|---|---|---|---|---|
| AXA | Ma Maison (habitation) CG | `axa.fr/content/dam/axa-fr-convergence/habitation/ipid/Ma_Maison_CG.pdf` | 112 | 56 292 | ✅ Adobe PDF Library 17 |
| AXA | Mon Auto CG (972115D) | `espaceclient.axa.fr/…/CG/AUTO/972115D.pdf` | 80 | 38 517 | ✅ |
| MAIF | Assurance habitation CG (M5202AHA) | `maif.fr/maiffr/documents/pdf/…/conditions-generales-assurance-habitation.pdf` | 114 | 52 746 | ✅ |
| MAIF | Raqvam CG | `maif.fr/content/documents/public/…/CGRaqvamMaif.pdf` | 36 | 23 384 | ✅ (iLovePDF) |
| Macif | Habitation CG v06/2025 | `macif.fr/files/live/sites/maciffr/files/conditions_generales_habitation/CG-Macif-Habitation.pdf` | 100 | 41 799 | ✅ |
| Macif | Auto CG | `macif.fr/files/live/sites/maciffr/files/conditions_generales_vehicules/CG_Auto.pdf` | 68 | 26 283 | ✅ |

**≈ 239 000 words across 6 documents** — a serious amount of authentic product language.

**Allianz is the weak link.** The publicly indexed `espaceclient.allianz.fr/pdf/02/d_allianz_habitation.pdf`
is only a **2-page, 1 188-word extract**, not a full CG; `allianz.fr` returns 403 to automated
fetches. Full Allianz CGs are not reliably exposed at stable public URLs. Traditional-network
insurers (Allianz, Generali, agent-distributed products) are generally worse than mutuals
(MAIF, Macif, MAAF) on this.

### 3.2 How cleanly would they parse? Well.

* **Every sampled PDF has a real text layer** — no OCR anywhere. `pdftotext -layout` works directly.
* Text quality is high. Macif habitation extracts as clean, accented, well-ordered French with an
  explicit numbered hierarchy (`8 - Comment sont indemnisés les dommages ?` → `8.1 Indemnisation de
  vos dommages…` → `8.3 Détermination…`), and page footers carrying
  `Macif Habitation - Conditions générales - Version 06/2025` — free document-level metadata.
* Structural markers vary by insurer: AXA uses `Article N – TITRE`; Macif/MAIF use decimal numbering.
  A single regex heading detector will **not** cover all four insurers — expect one profile per issuer.
* Known rough edges: multi-column layout bleed on some AXA pages (`-layout` merged a heading with
  adjacent body text), and *tableaux de garanties* which are semantically dense tables that flatten
  badly. Both are normal PDF-RAG problems, not blockers.
* AXA's "Ma Maison" CG also appends the mutual's *statuts* (`TITRE PREMIER - CONSTITUTION…`,
  `Article 8 – FONDS D'ÉTABLISSEMENT`) — strip that tail or it pollutes retrieval.

### 3.3 Copyright — honest answer: these are **not** openly licensed

**MAIF is explicit** ([mentions légales](https://www.maif.fr/annexes/mentions-legales), §Propriété intellectuelle):

> « Vous avez l'autorisation de consulter les données qu'il contient pour votre seule utilisation
> personnelle non commerciale »
>
> « aucun logo, texte, son, graphique ou image contenus dans le site ne pourrait être copié,
> reproduit, modifié, publié, émis, posté, transmis ou distribué par quelques moyens que ce soit
> sans l'autorisation préalable écrite de MAIF »

**AXA's** `mentions-legales.html` page contains **no** intellectual-property clause at all (verified
by fetching it). Silence is not permission — default French copyright applies.

**Macif and Allianz** legal pages return 403 to automated fetching, so their terms are
**unverified**. Assume the same posture.

**The one legal handle for local use** is the French TDM exception,
[CPI art. L122-5-3 III](https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000044363192):

> « Des copies ou reproductions numériques d'œuvres auxquelles il a été accédé de manière licite
> peuvent être réalisées en vue de fouilles de textes et de données menées à bien par toute personne,
> quelle que soit la finalité de la fouille, sauf si l'auteur s'y est opposé de manière appropriée,
> notamment par des procédés lisibles par machine pour les contenus mis à la disposition du public
> en ligne. »

But the **same article** adds:

> « Les copies et reproductions sont stockées avec un niveau de sécurité approprié puis détruites à
> l'issue de la fouille de textes et de données. »

**Read that honestly.** The exception plausibly covers *indexing these PDFs locally to build and
evaluate a RAG system*. It does **not** authorise republishing the PDFs or their extracted text, and
the destruction clause makes a *permanent, public* derived corpus legally shaky. I am not a lawyer
and this is a genuine grey zone — flagging it rather than asserting it is fine.

➡️ **Operational rule: never commit CG PDFs, extracted text, or a chunked/embedded derivative of
them to the public repo.** Ship a `sources.yaml` manifest of URLs + a downloader script, and
`.gitignore` everything it produces. That keeps the project reproducible without redistributing
anything.

---

## 4. ACPR and France Assureurs

### 4.1 ACPR — good content, unclear licence

**Volume.** The [registre officiel → Recommandations](https://acpr.banque-france.fr/fr/reglementation/registre-officiel/recommandations)
lists **17 recommandations** across 2 pages *(measured: 12 + 5)*. The insurance-relevant ones:

* `2024-R-03` (21 nov. 2024) — recueil des informations client, devoir de conseil et recommandation
  personnalisée **en assurance**
* `2024-R-02` (2 juil. 2024) — traitement des réclamations
* `2024-R-01` (28 juin 2024) — mise en œuvre de la directive (UE) 2016/97 (DDA)
* `2022-R-02` — caractéristiques extra-financières dans les communications publicitaires
* `2021-R-01` — contrats d'assurance-vie liés au financement en prévision d'obsèques
* `2019-R-01` — communications à caractère publicitaire des contrats d'assurance vie
* `2017-R-01` — libre choix de l'assurance emprunteur
* `2016-R-04` — commercialisation des contrats d'assurance vie en unités de compte

Beyond recommandations, the registre officiel also carries *Avis, Décisions, Instructions, Lignes
directrices, Listes, Notices, Positions, Principes d'application sectoriels* and the *Recueil des
sanctions* — the sanctions recueil in particular is a large body of reasoned French insurance-law
prose. Not enumerated here.

**Access.** Drupal site behind Akamai. *(measured)*: the default `WebFetch`/`curl` User-Agent gets
**403**; a normal desktop browser UA gets **200**. `robots.txt` itself is 403-gated. There is **no
public API**; you follow list pages (`?page=0`, `?page=1`) → publication detail pages
(`/fr/publications-et-statistiques/publications/…`) → PDFs at `/system/files/…`, which download fine.
Modest, polite scraping is workable; a bulk feed is not offered.

**Parse difficulty: easy.** Sampled `recommandation_2015-r-04.pdf` (48 KB): real text layer, single
column, numbered sections, produced from Word via Acrobat PDFMaker. Extracts near-perfectly.

**Licensing — flagged ambiguity.** The ACPR's
[mentions légales](https://acpr.banque-france.fr/fr/mentions-legales) (page last updated 2 March 2026,
fetched 2026-08-01) contain **no reuse or licence clause whatsoever** — only publisher and host
identification. There is no Etalab notice, unlike Légifrance or service-public.

The default rule would be [CRPA art. L321-1](https://www.legifrance.gouv.fr/codes/section_lc/LEGITEXT000031366350/LEGISCTA000031367685/):

> « Les informations publiques figurant dans des documents communiqués ou publiés par les
> administrations mentionnées au premier alinéa de l'article L. 300-2 peuvent être utilisées par
> toute personne qui le souhaite à d'autres fins que celles de la mission de service public pour les
> besoins de laquelle les documents ont été produits ou reçus. »

…but art. L321-2 c) carves out information « sur [laquelle] des tiers détiennent des droits de
propriété intellectuelle », and the ACPR is an independent authority backed by the Banque de France
rather than a plain-vanilla administration.

➡️ **Verdict: consume, don't redistribute.** Fetch ACPR PDFs at build time, index them locally, cite
them by URL. Do not commit the PDFs into the public repo. This is a real uncertainty, not a
formality — it would need a rights query to the ACPR to resolve.

### 4.2 France Assureurs — closed. Exclude.

[Mentions légales](https://www.franceassureurs.fr/mentions-legales/), verbatim:

> « Les contenus du site (commentaires, éditoriaux, illustrations, images, circulaires, notes,
> rapports, vidéos ou autres), ainsi que leur forme sont protégés par le droit d'auteur. »
>
> « Toute reproduction, représentation, diffusion ou autres utilisations des éléments du site, par
> quelque procédé que ce soit, est, sauf accord exprès de la Fédération française de l'assurance,
> strictement interdite. »
>
> « La reproduction, la représentation, la diffusion ou toute autre utilisation des documents et
> informations du site à des fins non lucratives **est autorisée, sur accord exprès** de La
> Fédération française de l'Assurance… »

Note the trap: even **non-commercial** reuse requires **express prior agreement**. That is stricter
than most French institutional sites and rules France Assureurs out for an unattended public repo.
Its output (*Données clés de l'assurance française*, annual reports, tableaux de bord) is statistical
anyway — market figures, not the kind of explanatory prose a RAG system answers questions from.

➡️ **Exclude.** Cite figures narratively if ever needed; do not ingest.

### 4.3 Adjacent, checked briefly

* **La Médiation de l'Assurance** — annual reports at
  [mediation-assurance.org/rapports-activite](https://www.mediation-assurance.org/rapports-activite/),
  free PDFs back to 2016. Content is genuinely valuable (real dispute typologies, worked cases,
  claim-handling doctrine). **Licence not verified** — it is an association-run scheme with no
  visible open licence. Treat like insurer material: link, don't redistribute.
* **ABE Info Service** (abe-infoservice.fr, joint ACPR/AMF consumer site) — legal page returned 403
  to automated fetch; **unverified**. Content overlaps heavily with service-public.fr anyway.
* **EUR-Lex** — worth adding for the European layer (Solvabilité II 2009/138/CE, DDA/IDD (UE) 2016/97).
  Per the [EUR-Lex legal notice](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html),
  reuse is governed by **Decision 2011/833/EU**; editorial content, legislative summaries and
  consolidated texts are **CC-BY-4.0**, and metadata is **CC0**. Redistribution is fine with
  attribution. Clean HTML/XML, French official version available.

---

## 5. data.gouv.fr — a dead end for text

*(measured)* — querying the catalogue API (`/api/1/datasets/?q=assurance`) returns **15 datasets**.
What is actually there:

* **Municipal procurement notices** — *"Souscription et gestion d'un marché d'assurance Risques
  Statutaires"* (Brignoles), *"Assurance Dommages aux Biens"* (Le Malesherbois), Guesnain, Antibes…
  These are tender records, not insurance content.
* **Health/social-insurance statistics** — DREES / Ministère des Solidarités: *"Couverture des risques
  sociaux par les organismes privés d'assurance"* (`lov2`), *"Complémentaire santé…"* (`fr-lo`),
  *"Les durées d'assurance validées par les actifs pour leur retraite"* (`fr-lo`). Openly licensed,
  but tabular — no prose to retrieve over.
* **Corporate ESG filings** — CNP Assurances fleet-emissions declarations (`notspecified`).

➡️ **No French insurance *text* corpus exists on data.gouv.fr.** The genuinely useful data.gouv
assets for this project are the **DILA ones already covered above** (the service-public flux and the
Légifrance dumps are catalogued there). Several municipal datasets are also `notspecified` licence,
which is another reason to skip them.

---

## 6. Summary of dead ends and open questions

**Dead ends (confirmed):**

* data.gouv.fr has no insurance text corpus — only tenders and statistics.
* France Assureurs is off-limits without a written agreement.
* Scraping legifrance.gouv.fr HTML is blocked (403 even with a browser UA).
* Allianz does not publish complete CGs at stable public URLs; `allianz.fr` blocks automated access.
* No public API exists for ACPR publications.

**Open questions I could not resolve — do not treat these as settled:**

1. **Légifrance/PISTE concrete rate limits.** Only "the provider sets limits at its discretion" is
   public. Numbers presumably appear inside the portal after registration.
2. **ACPR redistribution rights.** No licence published anywhere on their site. Resolving this needs
   a direct query to the ACPR.
3. **Macif / Allianz IP terms.** Their legal pages block automated fetch; read manually before
   assuming anything.
4. **La Médiation de l'Assurance licence.** Unverified.
5. **How much of `harvard-lil/cold-french-law` is Code des assurances.** The HF filter endpoint
   500'd; count locally after download.
6. **Whether TDM-exception copies may persist.** L122-5-3 III's destruction clause vs. a long-lived
   vector index is unsettled; this is exactly why insurer CGs stay out of the repo.

---

## RECOMMENDATION

### Tier 1 — the starter corpus. Openly licensed, commit it.

**1. service-public.fr insurance fiches — the backbone.**
79 documents / ≈ 102 k words of consumer-register insurance French, Licence Ouverte 2.0, one 23.5 MB
ZIP with no auth, XML that already carries chunk boundaries, topical breadcrumbs, modification dates
and **cross-links to the exact Légifrance sections each fiche rests on**. Nothing else in this survey
combines "clearly redistributable", "zero access friction", "pre-structured" and "written in the
register users actually ask questions in". Start here.

**2. Code des assurances (in force) — the authority layer.**
2 377 articles (L 937 / R 1 142 / A 246 / D 52), Licence Ouverte 2.0, ~3 MB. Pull it from
`louisbrulenaudet/code-assurances` (fastest, already API-shaped, has `fullSectionsTitre`) or filter
`harvard-lil/cold-french-law` on `texte_titre == "Code des assurances"` (cleaner CC-BY-4.0 story,
Markdown content, but ~1 GB download). Keep the **DILA bulk LEGI route documented** as the
reproducible, no-third-party path even if you don't run it day-to-day.

**3. Optional third layer — EUR-Lex Solvabilité II + DDA.** CC-BY-4.0, ~2 documents, adds the
European directive level that the Code des assurances transposes.

**Why this combination:** it gives the system *two registers of the same subject matter* —
"what a policyholder asks" (service-public) and "what the law says" (code) — already joined by the
`<dc:source>` links. That join is the interesting retrieval problem, and it costs nothing legally.
Total footprint ≈ **2 456 documents, ~5 MB, 100 % redistributable**, and the whole ingest is
reproducible from two URLs.

### Tier 2 — fetch at build time, `.gitignore` the artifacts, link in the docs.

**4. ACPR registre officiel** (17 recommandations + notices/positions/sanctions). Excellent
supervisory-doctrine French, near-perfect PDF extraction, requires only a browser User-Agent. But no
published licence → **do not commit the PDFs**. A `fetch_acpr.py` + URL manifest keeps the project
reproducible without redistributing anything.

### Tier 3 — local-only, opt-in, never committed.

**5. Insurer CGs (AXA, MAIF, Macif).** ≈ 239 k words of the real thing — the actual contractual
language a production insurance RAG would have to handle, with clean text layers. This is the most
*useful* material in the survey and the most *legally constrained*. MAIF forbids redistribution in
writing; AXA is silent (which is not permission); Macif/Allianz are unverified. CPI L122-5-3 III
plausibly covers local mining, but its "destroy the copies afterwards" clause makes a permanent
public derivative untenable. **Ship a downloader + manifest, gitignore the PDFs and every
derivative, and say so explicitly in the README.** That is the honest posture and it costs the
portfolio nothing — the demo still works on a reviewer's machine.

### Excluded

* **France Assureurs** — express prior agreement required even for non-commercial reuse.
* **data.gouv.fr insurance datasets** — no text; several have `notspecified` licences.
* **Scraping Légifrance HTML** — blocked, and unnecessary given the API and dumps.

### Suggested first implementation step

Build `ingest/` with two loaders — `service_public.py` (ZIP → XML → `Document` with
`{fiche_id, title, theme, dossier, modified, legifrance_refs, section_path}`) and
`code_assurances.py` (parquet → `Document` with `{article_num, etat, date_debut, section_path}`) —
and a `sources.yaml` recording, for each source, the **licence, the long download URL, the filename
and the file date**. Those last three are not bookkeeping: they are the attribution condition that
Licence Ouverte 2.0 imposes on both Tier-1 sources.

---

## Sources

**Légifrance / DILA / PISTE**
- <https://www.legifrance.gouv.fr/contenu/pied-de-page/foire-aux-questions-api>
- <https://www.legifrance.gouv.fr/contenu/pied-de-page/open-data-et-api>
- <https://www.legifrance.gouv.fr/loda/id/JORFTEXT000029135221> (Arrêté du 24 juin 2014)
- <https://www.dila.gouv.fr/home/open-data-et-api>
- <https://echanges.dila.gouv.fr/OPENDATA/LEGI/> and `DILA_LEGI_Presentation_20170824.pdf`
- <https://piste.gouv.fr/en/?format=raw&option=com_apiportal&view=terms> (PISTE CGU §3.1.3, §2.6)
- <https://piste.gouv.fr/images/cgu/DILA_Legifrance_Beta_v2.pdf>

**service-public.fr**
- <https://www.data.gouv.fr/datasets/fiches-pratiques-et-ressources-de-service-public-gouv-fr-particuliers>
- <https://lecomarquage.service-public.gouv.fr/vdd/3.5/part/zip/vosdroits-latest.zip>
- <https://echanges.dila.gouv.fr/OPENDATA/SERVICE-PUBLIC_DTD/schema_3.5.zip>
- <https://www.service-public.gouv.fr/a-propos/mentions-legales>

**Datasets**
- <https://huggingface.co/datasets/louisbrulenaudet/code-assurances> (+ `datasets-server.huggingface.co` `/info`, `/size`, `/rows`, `/filter`)
- <https://huggingface.co/datasets/harvard-lil/cold-french-law>

**Insurers**
- <https://www.axa.fr/content/dam/axa-fr-convergence/habitation/ipid/Ma_Maison_CG.pdf>
- <https://espaceclient.axa.fr/content/dam/axa/ecc/pdf/Espace%20Documentaire/CG/AUTO/972115D.pdf>
- <https://www.maif.fr/maiffr/documents/pdf/documentation-contractuelle/habitation/conditions-generales-assurance-habitation.pdf>
- <https://www.maif.fr/content/documents/public/maif-fr/pdf/contrat/pp/CGRaqvamMaif.pdf>
- <https://www.macif.fr/files/live/sites/maciffr/files/conditions_generales_habitation/CG-Macif-Habitation.pdf>
- <https://www.macif.fr/files/live/sites/maciffr/files/conditions_generales_vehicules/CG_Auto.pdf>
- <https://www.maif.fr/annexes/mentions-legales>
- <https://www.axa.fr/mentions-legales.html>

**Regulators / federation / law**
- <https://acpr.banque-france.fr/fr/reglementation/registre-officiel/recommandations>
- <https://acpr.banque-france.fr/fr/mentions-legales>
- <https://www.franceassureurs.fr/mentions-legales/>
- <https://www.mediation-assurance.org/rapports-activite/>
- <https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000044363192> (CPI L122-5-3)
- <https://www.legifrance.gouv.fr/codes/section_lc/LEGITEXT000031366350/LEGISCTA000031367685/> (CRPA L321-1, L321-2)
- <https://eur-lex.europa.eu/content/legal-notice/legal-notice.html>

**data.gouv.fr**
- <https://www.data.gouv.fr/api/1/datasets/?q=assurance>
