
# Load required libraries
library(dplyr)
library(arrow)
library(googleCloudStorageR)
library(tidyverse)
library(reticulate)
source("src/func.R")

# setup python
use_python("~/scmcp/bin/python3", required = TRUE)

os = import("os")
pd = import("pandas")
ad = import("anndata")
gcsfs = import("gcsfs")

gcs_base_path = "gs://arc-ctc-scbasecamp/2025-02-25"
feature_type = "GeneFull_Ex50pAS"
#meta data
#gcloud storage cp gs://arc-ctc-scbasecamp/2025-02-25/metadata/GeneFull_Ex50pAS/Homo_sapiens/sample_metadata.parquet ./


sample_meta <- read_parquet("Tahoe100/data_2025/sample_metadata.parquet")

panc_sample_meta <- sample_meta %>% dplyr::filter(tissue == "pancreas", disease == "insulin resistance")

obs_meta <- read_parquet("Tahoe100/data_2025/obs_metadata.parquet")

panc_obs_meta <- obs_meta %>% dplyr::filter(SRX_accession %in% panc_sample_meta$srx_accession)

h5ad_example <- panc_obs_meta %>% dplyr::filter(SRX_accession == "SRX16005379")
