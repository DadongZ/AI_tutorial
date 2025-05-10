
# helper function
get_parquet_files = function(gcs_base_path, feature_type, target = NULL, endswith = NULL) {
  files_glob <- fs$glob(paste0(gcs_base_path, "/", feature_type, "/**"))
  
  if (!is.null(target)) {
    files = files[sapply(files, function(f) basename(f) == target)]
  } else if (!is.null(endswith)) {
    files = files[sapply(files, function(f) grepl(paste0(endswith, "$"), f))]
  }
  
  file_list = lapply(files, function(f) {
    parts = unlist(strsplit(f, "/"))
    c(parts[length(parts)-1], f)
  })
  
  file_df = as.data.frame(do.call(rbind, file_list), stringsAsFactors = FALSE)
  colnames(file_df) = c("organism", "file_path")
  
  return(file_df)
}

# function to read parquet files from GCS
read_data = function(file, n = NULL){
  if(!startsWith(file, "gs://")){
    file = paste0("gs://", file)
  }
  dataset = open_dataset(file, format = "parquet")
  if (is.null(n)) {
    dataset %>% collect() %>% as.data.frame()
  } else {
    dataset %>% head(n) %>% collect() %>% as.data.frame()
  }
}