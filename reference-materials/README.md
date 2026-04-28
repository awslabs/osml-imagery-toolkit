# Reference Materials — osml-imagery-toolkit

This directory contains authoritative standards and specifications for the imagery formats, sensor models, and image processing algorithms implemented by this library. When questions arise about algorithm correctness, field interpretation, projection equations, or processing requirements — consult the specs here rather than relying on assumptions or training data.

For a full list of available documents, download URLs, and directory structure, see [CATALOG.md](CATALOG.md).

## How to Use These Specs (Agent Navigation Strategy)

These PDFs range from 6 to 435 pages. They are too large to read in full. Use the following process to find authoritative answers efficiently:

### Step 1: Identify the Right Document

Check the source code you're working in for spec references — packages and modules may include pointers to governing specifications in their docstrings or comments. These references are being added incrementally so they won't always be present. When they are absent, use the topic index below to identify candidate documents.

### Step 2: Read the Table of Contents

For any document over ~30 pages, start by reading pages 1–5 (which typically contain the TOC and list of figures/tables). This tells you which section to target.

```
Read the PDF with pages="1-5" to get the TOC
```

### Step 3: Navigate to the Relevant Section

Once you've identified the section from the TOC, read just those pages. For example, if the TOC says "Section 5.3 — IGEOLO" starts on page 47, read pages 47–52.

### Step 4: Keyword Search as Fallback

If the TOC doesn't clearly map to your question (e.g., looking for a specific field name like `COLLECTDG` or `CMETAA`), try reading pages in small ranges and scanning for the term. Many of these specs have alphabetical appendices or field-definition tables you can search through.

### Authority Hierarchy

When multiple specs cover the same topic, precedence is:

1. **SIPS** — authoritative for image processing algorithms
2. **SICD/SIDD volumes** — authoritative for SAR sensor model math and format specifics
3. **STDI-0002** appendices — authoritative for TRE/DES field interpretation
4. **JBP** (Joint BIIF Profile) — authoritative for NITF structure
5. **MIL-STD-2500A/B/C** — historical reference only; defines older NITF 2.0 format

---

## Topic Index

This project is focused on **image processing algorithms** and **photogrammetry/sensor models**. The topic index is organized around the questions that arise in this context. For file-format parsing questions (byte offsets, field sizes, encoding), see the osml-imagery-io project's reference guide.

### Image Processing Algorithms

This is the primary spec domain for this project. The SIPS standard defines the algorithms implemented in the `image_processing` package.

**"What resampling kernel or method should be used for reduced resolution datasets?"**
→ `SIPS/NGA_STD_0014_2.5_SIPS.pdf` (138 pp) — TOC on pages 1–3. Look for sections on RRDS generation and interpolation.

**"What DRA (dynamic range adjustment) algorithm is standard?"**
→ `SIPS/NGA_STD_0014_2.5_SIPS.pdf` — DRA and histogram-based display sections.

**"How should image pyramids be built (filtering, decimation, antialiasing)?"**
→ `SIPS/NGA_STD_0014_2.5_SIPS.pdf` — RRDS (Reduced Resolution Data Set) generation sections.

**"What orthorectification approach is standard?"**
→ `SIPS/NGA_STD_0014_2.5_SIPS.pdf` — orthorectification and terrain correction sections.

**"How should complex SAR pixels (I/Q) be converted to magnitude for display?"**
→ `SIPS/SAND2015-2309.pdf` (30 pp) — complex pixel representations and domain transforms.

**"What scaling or radiometric calibration applies to SAR display?"**
→ `SIPS/SAND2019-2371.pdf` (48 pp) — scaling, dynamic range, and radiometric calibration for SAR.

### Sensor Models — Projection Equations

These specs define the math for converting between image coordinates and geodetic positions.

**"How does the SICD image-to-ground projection work (PFA, RMA, RGZCOMP, INCA)?"**
→ `SICD/NGA.STND.0024-3_1.3.0_SICD_IPDD_FINAL.pdf` (122 pp) — the complete projection equations for all SICD image formation types. TOC on pages 1–3.

**"How does the SIDD measurement projection work?"**
→ `SIDD/NGA.STND.0025-1_3.0_SIDD_DIDD.pdf` (93 pp) — Section on Measurement and ExploitationFeatures. TOC on pages 1–3.

**"How do RPC (rational polynomial coefficient) sensor models work mathematically?"**
→ `SensorModels/cubic-rational-polynomial-camera-model.pdf` (11 pp) — the Hartley & Saxena theory paper. Short enough to read in full.

**"How does the Replacement Sensor Model (RSM) work conceptually?"**
→ `SensorModels/the-replacement-sensor-model-rsm-overview-status-and-28yzvmuhlk.pdf` (10 pp) — overview paper. Short enough to read in full.

**"What is the formal RSM polynomial structure, ground domain, normalization?"**
→ `JBP/STDI-0002-v2025.2-202601/Vol1-AppU-RSM_202207.pdf` (295 pp) — the complete RSM TRE specification. TOC on pages 1–6. Key sections: polynomial definitions, ground domain forms, sectioned polynomials.

### Sensor Models — TRE Field Interpretation

When the toolkit receives parsed TRE data from osml-imagery-io, these specs explain what the field values *mean* and how to use them in computations.

**"What do the RPC00B coefficient fields represent? How are they used in the projection?"**
→ `JBP/STDI-0002-v2025.2-202601/Vol1-AppC-PIAE_202506.pdf` — RPC TRE definitions with field semantics and normalization conventions.

**"How do ICHIPB fields describe the chip-to-full-image coordinate transform?"**
→ `JBP/STDI-0002-v2025.2-202601/Vol1-AppB-ICHIPB_202410.pdf` (30 pp) — transformation equations and field interpretation.

**"What do SENSRB fields tell me about sensor geometry?"**
→ `JBP/STDI-0002-v2025.2-202601/Vol1-AppZ-SENSRB_202506.pdf` (122 pp) — sensor parameters, pointing, timing, and uncertainty.

**"What geospatial positioning data is in GEOSDE TREs (GEOLOB, GEOSDTA)?"**
→ `JBP/STDI-0002-v2025.2-202601/Vol1-AppP-GEOSDE_202404.pdf` (435 pp) — TOC on pages 1–5. Contains multiple TRE definitions related to geolocation.

**"How do BANDSB fields describe spectral bands and calibration?"**
→ `JBP/STDI-0002-v2025.2-202601/Vol1-AppX-BANDSB_202502.pdf` (38 pp)

**"What attitude/position data is in ATTPTA?"**
→ `JBP/STDI-0002-v2025.2-202601/Vol1-AppW-ATTPTA_202502.pdf` (36 pp)

### Elevation Models

**"How is DTED formatted and what are the post spacing levels?"**
→ `DTED/MIL-PRF-89020B.pdf` (45 pp) — DTED performance specification including data structure.

**"What is the HRE profile for elevation products?"**
→ `HRE/HRE Product IP v1 1 FINAL.pdf`

### GeoTIFF Georeferencing

**"How does GeoTIFF georeferencing work (GeoKeys, ModelTiepoints, transformation tags)?"**
→ `GeoTIFF/OGCGeoTIFFStandard.pdf` (112 pp) — OGC GeoTIFF 1.1. TOC on page 3.

**"What is Cloud Optimized GeoTIFF and how does tiling/overview structure work?"**
→ `GeoTIFF/OGCCloudOptimizedGeoTIFFStandard.pdf` (34 pp)

### SICD/SIDD Metadata Interpretation

**"What XML metadata fields does SICD define and what do they mean?"**
→ `SICD/NGA.STND.0024-1_1.3.0_SICD_DIDD_FINAL.pdf` (183 pp) — design, XML schema documentation, and field semantics. TOC on pages 1–4.

**"How is a SICD NITF file formatted?"**
→ `SICD/NGA.STND.0024-2_1.3.0_SICD_FFDD_FINAL.pdf` (38 pp)

**"How is SIDD stored in NITF or GeoTIFF?"**
→ `SIDD/NGA.STND.0025-2_3.0_SIDD_NITF_FFDD.pdf` (47 pp) or `SIDD/NGA.STND.0025-3_3.0-SIDD_GEOTIFF.pdf` (17 pp)

---

## TRE/DES Quick Lookup

When you need the definition of a specific TRE or DES, use this table to find the exact file. These appendices define field semantics — the interpretation of values that have already been parsed from the binary format by osml-imagery-io.

| TRE/DES Name | File |
|---|---|
| ICHIPB | `STDI-0002-v2025.2-202601/Vol1-AppB-ICHIPB_202410.pdf` |
| PIAE (RPC/RPC00B) | `STDI-0002-v2025.2-202601/Vol1-AppC-PIAE_202506.pdf` |
| CSDE (CSEPHB) | `STDI-0002-v2025.2-202601/Vol1-AppD-CSDE_202502.pdf` |
| ASDE (ACFTB, AIMIDB, etc.) | `STDI-0002-v2025.2-202601/Vol1-AppE-ASDE_202502.pdf` |
| IOMAPA | `STDI-0002-v2025.2-202601/Vol1-AppF-IOMAPA_202110.pdf` |
| NBLOCA | `STDI-0002-v2025.2-202601/Vol1-AppI-NBLOCA_202110.pdf` |
| HISTOA | `STDI-0002-v2025.2-202601/Vol1-AppL-HISTOA_202110.pdf` |
| ENGRDA | `STDI-0002-v2025.2-202601/Vol1-AppN-ENGRDA_202110.pdf` |
| MITOCA | `STDI-0002-v2025.2-202601/Vol1-AppO-MITOCA_202110.pdf` |
| GEOSDE (GEOLOB, GEOSDTA, etc.) | `STDI-0002-v2025.2-202601/Vol1-AppP-GEOSDE_202404.pdf` |
| NSDE (RSMIDA, RSMPCA, etc.) | `STDI-0002-v2025.2-202601/Vol1-AppR-NSDE_202410.pdf` |
| RSM (full spec) | `STDI-0002-v2025.2-202601/Vol1-AppU-RSM_202207.pdf` |
| DPPDB | `STDI-0002-v2025.2-202601/Vol1-AppV-DPPDB_202110.pdf` |
| ATTPTA (attitude/position) | `STDI-0002-v2025.2-202601/Vol1-AppW-ATTPTA_202502.pdf` |
| BANDSB | `STDI-0002-v2025.2-202601/Vol1-AppX-BANDSB_202502.pdf` |
| J2KLRA, J2KLRB | `STDI-0002-v2025.2-202601/Vol1-AppY-J2KLRA-J2KLRB_202407.pdf` |
| SENSRB | `STDI-0002-v2025.2-202601/Vol1-AppZ-SENSRB_202506.pdf` |
| PIXQLA | `STDI-0002-v2025.2-202601/Vol1-AppAA-PIXQLA_202210.pdf` |
| RELCCA | `STDI-0002-v2025.2-202601/Vol1-AppAD-RELCCA_202310.pdf` |
| XMLDCA | `STDI-0002-v2025.2-202601/Vol1-AppAE-XMLDCA_202110.pdf` |
| MIE4NITF | `STDI-0002-v2025.2-202601/Vol1-AppAF-MIE4NITF_202506.pdf` |
| CCINFA | `STDI-0002-v2025.2-202601/Vol1-AppAG-CCINFA_202506.pdf` |
| GLAS-GFM | `STDI-0002-v2025.2-202601/Vol1-AppAH-GLAS-GFM_202110.pdf` |
| SECURA | `STDI-0002-v2025.2-202601/Vol1-AppAI-SECURA_202410.pdf` |
| PIXMTA | `STDI-0002-v2025.2-202601/Vol1-AppAJ-PIXMTA_202506.pdf` |
| MATESA | `STDI-0002-v2025.2-202601/Vol1-AppAK-MATESA_202506.pdf` |
| ILLUMA, ILLUMB | `STDI-0002-v2025.2-202601/Vol1-AppAL-ILLUMA-ILLUMB_202504.pdf` |
| FRMSGA | `STDI-0002-v2025.2-202601/Vol1-AppAN-FRMSGA_202506.pdf` |
| SODDXA | `STDI-0002-v2025.2-202601/Vol1-AppAP-SODDXA_202504.pdf` |
| ASTORA | `STDI-0002-v2025.2-202601/Vol1-AppAQ-ASTORA_202204.pdf` |
| BCHIPA | `STDI-0002-v2025.2-202601/Vol1-AppAR-BCHIPA_202404.pdf` |
| CSDIDA, SYSIDA | `STDI-0002-v2025.2-202601/Vol1-AppAS-CSDIDA-SYSIDA_202502.pdf` |
| S2EVPA | `STDI-0002-v2025.2-202601/Vol1-AppAT-S2EVPA_202506.pdf` |
| COMNTA | `STDI-0002-v2025.2-202601/Vol1-AppAU-COMNTA_202306.pdf` |
| CSCCGA | `STDI-0002-v2025.2-202601/Vol1-AppAV-CCIS-CSCCGA_202406.pdf` |
| CSCRNA, FCRNSA | `STDI-0002-v2025.2-202601/Vol1-AppAW-CSCRNAandFCRNSA_202404.pdf` |
| ISAR | `STDI-0002-v2025.2-202601/Vol1-AppAX-ISAR_202402.pdf` |
| SORBXA | `STDI-0002-v2025.2-202601/Vol1-AppAY-SORBXA_202504.pdf` |
| TRE Overflow DES | `STDI-0002-v2025.2-202601/Vol2-AppA-TREOverflow_202110.pdf` |
| CSATTA | `STDI-0002-v2025.2-202601/Vol2-AppC-CSATTA_202110.pdf` |
| CSSHPA, CSSHPB | `STDI-0002-v2025.2-202601/Vol2-AppD-CSSHPA-CSSHPB_202506.pdf` |
| GLAS-GFM DES | `STDI-0002-v2025.2-202601/Vol2-AppM-GLAS-GFM_202505.pdf` |
| XML_DATA_CONTENT | `STDI-0002-v2025.2-202601/Vol2-AppF-XML_DATA_CONTENT_202401.pdf` |

If a TRE name isn't in this table, check the STDI-0002 file listing — appendix filenames contain the extension name.
