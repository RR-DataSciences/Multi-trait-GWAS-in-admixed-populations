# --- AUTO-GENERATED ENVIRONMENT SETUP ---

Sys.setenv(WORKSPACE_CDR = "wb-silky-artichoke-2408.C2024Q3R9")
Sys.setenv(GOOGLE_CLOUD_PROJECT = "wb-perky-cabbage-8342")
Sys.setenv(GOOGLE_PROJECT = "wb-perky-cabbage-8342")
Sys.setenv(WORKSPACE_BUCKET = "gs://workspace-bucket-wb-perky-cabbage-8342")
Sys.setenv(WORKSPACE_TEMP_BUCKET = "gs://temporary-workspace-bucket-wb-perky-cabbage-8342")
Sys.setenv(EXPORT_BUCKET = "gs://workspace-bucket-wb-perky-cabbage-8342")
Sys.setenv(bucket_aou_tutorial = "NOT_FOUND")
Sys.setenv(bucket_id_aou_tutorial = "NOT_FOUND")
Sys.setenv(bucket_migrated = "gs://rw-migration-aou-rw-f7a4d148")
Sys.setenv(bucket_id_migrated = "rw-migration-aou-rw-f7a4d148")

# --- VERIFICATION BLOCK ---

vars_to_check <- c("WORKSPACE_CDR", "GOOGLE_CLOUD_PROJECT", "GOOGLE_PROJECT", "WORKSPACE_BUCKET", "WORKSPACE_TEMP_BUCKET", "EXPORT_BUCKET", "bucket_aou_tutorial", "bucket_id_aou_tutorial", "bucket_migrated", "bucket_id_migrated")

cat("\n🔍 Current Workspace Variables:\n")
for (v in vars_to_check) {
  value <- Sys.getenv(v)
  # Prints the name (padded to 22 chars) and the value
  cat(sprintf("  %-22s : %s\n", v, value))
}
message("\n✅ Environment Loaded Successfully.")
