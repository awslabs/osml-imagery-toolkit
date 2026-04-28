# Reference Materials Catalog

This catalog lists the authoritative standards and specifications relevant to satellite/UAV imagery processing, photogrammetry, and geospatial data formats. These documents are controlled by third parties and are not checked into the repository. Download the PDFs and place them in the appropriate subdirectory as shown in the directory structure below.

Most NGA documents are available from the [NGA Standards Registry](https://nsgreg.nga.mil/). The registry requires solving a CAPTCHA for access; automated tools cannot download directly.

## Directory Structure

```
reference-materials/
├── CATALOG.md                         # This file
├── README.md                          # Project-specific navigation guide
├── DTED/                              # Digital Terrain Elevation Data
│   ├── MIL-PRF-89020B.pdf
│   └── MIL-PRF-89020B Note1.pdf
├── GeoTIFF/                           # TIFF and GeoTIFF specifications
│   ├── TIFF6.pdf
│   ├── OGCGeoTIFFStandard.pdf
│   └── OGCCloudOptimizedGeoTIFFStandard.pdf
├── HRE/                               # High Resolution Elevation
│   └── HRE Product IP v1 1 FINAL.pdf
├── JBP/                               # NITF format and profiles
│   ├── Joint-BIIF-Profile-V2024.1_2024-01-18.pdf
│   ├── MIL-STD-188-199.pdf
│   ├── MIL-STD-2500A.pdf
│   ├── MIL-STD-2500A-CN1.pdf
│   ├── MIL-STD-2500B.pdf
│   ├── MIL-STD-2500B-CN1.pdf
│   ├── MIL-STD-2500B-CN2.pdf
│   ├── MIL-STD-2500C.pdf
│   ├── MIL-STD-2500C-CN1.pdf
│   ├── MIL-STD-2500C-CN2.pdf
│   ├── NCDRD_18February2010.pdf
│   ├── NGA.IP.0002_1.0 HRE.pdf
│   ├── NGA.STND.0044_1.3.3_MIE4NITF_202601.pdf
│   ├── NITF_MIL_STD_2500a.pdf
│   ├── STDI-0002-v2025.2-202601/      # TRE and DES definitions (multi-file)
│   └── USAF SARzip Standard V1.0.0.pdf
├── SensorModels/                      # Sensor model theory papers
│   ├── cubic-rational-polynomial-camera-model.pdf
│   └── the-replacement-sensor-model-rsm-overview-status-and-28yzvmuhlk.pdf
├── SICD/                              # Sensor Independent Complex Data
│   ├── NGA.STND.0024-1_1.3.0_SICD_DIDD_FINAL.pdf
│   ├── NGA.STND.0024-2_1.3.0_SICD_FFDD_FINAL.pdf
│   └── NGA.STND.0024-3_1.3.0_SICD_IPDD_FINAL.pdf
├── SIDD/                              # Sensor Independent Derived Data
│   ├── NGA.STND.0025-1_3.0_SIDD_DIDD.pdf
│   ├── NGA.STND.0025-2_3.0_SIDD_NITF_FFDD.pdf
│   └── NGA.STND.0025-3_3.0-SIDD_GEOTIFF.pdf
└── SIPS/                              # Softcopy Image Processing Standard
    ├── NGA_STD_0014_2.5_SIPS.pdf
    ├── SAND2015-2309.pdf
    └── SAND2019-2371.pdf
```

---

## NITF Standards

The National Imagery Transmission Format (NITF) is now governed by the Joint BIIF Profile (JBP), which replaced MIL-STD-2500C as the authoritative standard for NITF interoperability across NATO nations. The older MIL-STD documents are retained as historical reference for understanding legacy NITF 2.0/2.1 implementations.

### Joint BIIF Profile (JBP) — Current Standard

| File | Document | Source |
|------|----------|--------|
| `Joint-BIIF-Profile-V2024.1_2024-01-18.pdf` | ISO/IEC Joint BIIF Profile v2024.1 | [NGA i=5258](https://nsgreg.nga.mil/doc/view?i=5258) |

### STDI-0002 — Standard Data Extensions

The comprehensive specification for Tagged Record Extensions (TREs), Data Extension Segments (DESs), and implementation profiles. This is a multi-file package.

| File | Document | Source |
|------|----------|--------|
| `STDI-0002-v2025.2-202601/` (directory) | STDI-0002 v2025.2 — Full SDE package | [NGA i=5675](https://nsgreg.nga.mil/doc/view?i=5675) |

Key files within:
- `STDI-0002-SDE-Fundamentals-MainBody-V2025-2_202601.pdf` — Core fundamentals
- `STDI-0002-Volume-1-TREs-V2025-2_202601.pdf` — TRE definitions overview
- `STDI-0002-Volume-2-DESs-and-DESs-TREs-Combinations-V2025-2_202601.pdf` — DES definitions overview
- `STDI-0002-Volume-3-SDE-Profiles-and-Implementation-Guidance-V2025-2_202601.pdf` — Profiles
- `Vol1-AppU-RSM_202207.pdf` — Replacement Sensor Model TRE specification
- Individual appendices for each TRE/DES (SENSRB, BANDSB, ICHIPB, etc.)

### Historical MIL-STD References

| File | Document | Source |
|------|----------|--------|
| `NITF_MIL_STD_2500a.pdf` | MIL-STD-2500A — NITF Version 2.0 (1994) | [DLA QuickSearch](https://quicksearch.dla.mil/qsDocDetails.aspx?ident_number=112606) |
| — | MIL-STD-2500C — NITF Version 2.1 (superseded by JBP) | [NGA i=2063](https://nsgreg.nga.mil/doc/view?i=2063) |
| — | MIL-STD-2500C Change Notice 1 | [NGA i=4324](https://nsgreg.nga.mil/doc/view?i=4324) |
| — | MIL-STD-2500C Change Notice 2 | [NGA i=4724](https://nsgreg.nga.mil/doc/view?i=4724) |
| `MIL-STD-188-199.pdf` | VQ Decompression for NITF (1994) | [NGA i=2057](https://nsgreg.nga.mil/doc/view?i=2057) |

---

## NGA Implementation Profiles and Extensions

| File | Document | Source |
|------|----------|--------|
| `NGA.STND.0044_1.3.3_MIE4NITF_202601.pdf` | Motion Imagery Extension for NITF v1.3.3 | [NGA i=5676](https://nsgreg.nga.mil/doc/view?i=5676) |
| `NCDRD_18February2010.pdf` | NITF 2.1 Commercial Dataset Requirements Document (STDI-0006) | [Maxar CSDA](https://csda-maxar-pdfs.s3.amazonaws.com/NCDRD_18February2010.pdf) |

---

## SAR Imagery Standards

SAR (Synthetic Aperture Radar) imagery uses specialized NITF-based formats. Both SICD and SIDD files are NITF containers following specific guidelines defined in the JBP.

### SICD — Sensor Independent Complex Data (v1.3.0; 2021-11-30)

Standard for complex SAR imagery (Single Look Complex / Level 1 data).

| File | Document | Source |
|------|----------|--------|
| `NGA.STND.0024-1_1.3.0_SICD_DIDD_FINAL.pdf` | Vol 1 — Design & Implementation (183 pp) | [NGA i=5381](https://nsgreg.nga.mil/doc/view?i=5381) |
| `NGA.STND.0024-2_1.3.0_SICD_FFDD_FINAL.pdf` | Vol 2 — File Format (38 pp) | [NGA i=5382](https://nsgreg.nga.mil/doc/view?i=5382) |
| `NGA.STND.0024-3_1.3.0_SICD_IPDD_FINAL.pdf` | Vol 3 — Image Projections (122 pp) | [NGA i=5383](https://nsgreg.nga.mil/doc/view?i=5383) |
| — | XML Schema | [NGA i=5418](https://nsgreg.nga.mil/doc/view?i=5418) |

### SIDD — Sensor Independent Derived Data (v3.0; 2021-11-30)

Standard for derived SAR products (detected imagery, etc.).

| File | Document | Source |
|------|----------|--------|
| `NGA.STND.0025-1_3.0_SIDD_DIDD.pdf` | Vol 1 — Design & Implementation (93 pp) | [NGA i=5440](https://nsgreg.nga.mil/doc/view?i=5440) |
| `NGA.STND.0025-2_3.0_SIDD_NITF_FFDD.pdf` | Vol 2 — NITF File Format (47 pp) | [NGA i=5441](https://nsgreg.nga.mil/doc/view?i=5441) |
| `NGA.STND.0025-3_3.0-SIDD_GEOTIFF.pdf` | Vol 3 — GeoTIFF File Format (17 pp) | [NGA i=5442](https://nsgreg.nga.mil/doc/view?i=5442) |
| — | XML Schema | [NGA i=5231](https://nsgreg.nga.mil/doc/view?i=5231) |

### SARzip — SAR Compression Standard

| File | Document | Source |
|------|----------|--------|
| `USAF SARzip Standard V1.0.0.pdf` | USAF.RDUCE-001 — SAR Compression (SARzip) v1.0.0 (2017) | [NGA i=4506](https://nsgreg.nga.mil/doc/view?i=4506) |

---

## Image Processing

### SIPS — Softcopy Image Processing Standard (v2.5; 2026-05-20)

NGA standard defining image processing algorithms for softcopy exploitation across all sensor types (EO, IR, SAR, MSI, etc.). Covers resampling, dynamic range adjustment, orthorectification, sharpening, and display preparation.

| File | Document | Source |
|------|----------|--------|
| `NGA_STD_0014_2.5_SIPS.pdf` | NGA.STND.0014 v2.5 — Softcopy Image Processing Standard (138 pp) | [NGA i=5732](https://nsgreg.nga.mil/doc/view?i=5732) |

### SAR-Specific Processing Reports (Sandia National Laboratories)

These Sandia reports complement the SIPS standard with SAR-specific image processing guidance that SIPS does not cover, such as complex pixel handling and radiometric calibration for SAR sensors.

| File | Document | Source |
|------|----------|--------|
| `SAND2015-2309.pdf` | SAR Image Complex Pixel Representations (Doerry, 2015) (30 pp) | [OSTI](https://www.osti.gov/servlets/purl/1177594) |
| `SAND2019-2371.pdf` | SAR Image Scaling, Dynamic Range, Radiometric Calibration, and Display (Doerry, 2019) (48 pp) | [OSTI](https://www.osti.gov/servlets/purl/1761879) |

---

## GeoTIFF Standards

| File | Document | Source |
|------|----------|--------|
| `TIFF6.pdf` | TIFF Revision 6.0 — Base format specification (121 pp) | [ITU](https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf) |
| `OGCGeoTIFFStandard.pdf` | OGC GeoTIFF Standard v1.1 (19-008r4) (112 pp) | [OGC](https://www.ogc.org/standard/geotiff/) |
| `OGCCloudOptimizedGeoTIFFStandard.pdf` | OGC Cloud Optimized GeoTIFF v1.0 (21-026) (34 pp) | [OGC](https://www.ogc.org/standards/ogc-cloud-optimized-geotiff/) |

---

## Elevation Data

### DTED — Digital Terrain Elevation Data

The standard format for gridded elevation data used by DoD systems. DTED files provide terrain elevation post spacing at various resolutions (Level 0 through Level 2).

| File | Document | Source |
|------|----------|--------|
| `MIL-PRF-89020B.pdf` | MIL-PRF-89020B — DTED Performance Specification (45 pp) | [NGA i=2126](https://nsgreg.nga.mil/doc/view?i=2126) |
| `MIL-PRF-89020B Note1.pdf` | MIL-PRF-89020B Notice 1 (Amendment) | [NGA i=2126](https://nsgreg.nga.mil/doc/view?i=2126) |

### HRE — High Resolution Elevation

Profiles for raster elevation data products in NITF and GeoTIFF containers.

| File | Document | Source |
|------|----------|--------|
| `NGA.IP.0002_1.0 HRE.pdf` | Implementation Profile for HRE Products v1.0 (2009) | [NGA i=4102](https://nsgreg.nga.mil/doc/view?i=4102) |
| `HRE Product IP v1 1 FINAL.pdf` | HRE Product Implementation Profile v1.1 | [NGA i=4102](https://nsgreg.nga.mil/doc/view?i=4102) |

---

## Sensor Model Theory

Background papers on the rational polynomial and replacement sensor model approaches used in photogrammetry.

| File | Document | Source |
|------|----------|--------|
| `cubic-rational-polynomial-camera-model.pdf` | The Cubic Rational Polynomial Camera Model (Hartley & Saxena, 2001) (11 pp) | [ANU](https://users.cecs.anu.edu.au/~hartley/Papers/cubic/cubic.pdf) |
| `the-replacement-sensor-model-rsm-overview-status-and-28yzvmuhlk.pdf` | The Replacement Sensor Model (RSM): Overview, Status, and Performance Summary (Dolloff et al., ASPRS 2008) (10 pp) | [SciSpace](https://scispace.com/pdf/the-replacement-sensor-model-rsm-overview-status-and-28yzvmuhlk.pdf) |

The formal RSM TRE specification is in `JBP/STDI-0002-v2025.2-202601/Vol1-AppU-RSM_202207.pdf`.
