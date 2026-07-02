#!/usr/bin/env python
# coding: utf-8

# # Load Environment Variables in Python and test
# 
# The following function reads your `~/.bashrc` file and sets each variable in Python’s runtime environment (`os.environ`), so you can use them directly in your code.
# 
# - This notebook also creates an R file `Setting_Env_Variables_p2.R` for R users including Rstudio users.
# - This notebook also creates an R file `Setting_Env_Variables_p2.sas` for SAS users. 
# 
# Any other notebook can reuse this function to reload environment variables at the start.

# In[ ]:


import os

with open(os.path.expanduser("~/.bashrc"), 'r') as f:
    for line in f:
        if line.strip().startswith('export '):
            parts = line.strip().replace('export ', '').split('=', 1)
            if len(parts) == 2:
                var_name = parts[0].strip()
                var_value = parts[1].strip().strip("'\"")
                
                # SKIP PATH completely!
                if var_name == 'PATH':
                    continue  # Skip this line
                    
                os.environ[var_name] = var_value


# In[ ]:


# Now you can access them directly:
print(f"WORKSPACE_CDR = {os.environ.get('WORKSPACE_CDR')}")
print(f"WORKSPACE_BUCKET = {os.environ.get('WORKSPACE_BUCKET')}")
print(f"GOOGLE_PROJECT = {os.environ.get('GOOGLE_PROJECT')}")


# In[ ]:


# !echo $WORKSPACE_CDR


# In[ ]:


# !echo $WORKSPACE_TEMP_BUCKET


# In[ ]:


import os

# Define the variables to capture from your Notebook VM
vars_to_capture = [
    "WORKSPACE_CDR", 
    "GOOGLE_CLOUD_PROJECT", 
    "GOOGLE_PROJECT", 
    "WORKSPACE_BUCKET", 
    "WORKSPACE_TEMP_BUCKET",
    "EXPORT_BUCKET",
    "bucket_aou_tutorial",     # Path (gs://...)
    "bucket_id_aou_tutorial",  # ID only (aou-tutorial-notebooks)
    "bucket_migrated",         # Path (gs://...)
    "bucket_id_migrated"       # ID only (rw-migration-xxxx)
]

r_file_name = "Setting_Env_Variables_p2.R"

# --- DYNAMIC HOME-BASED PATH RESOLUTION ---
# os.path.expanduser("~") automatically resolves to /home/jupyter, /home/dataproc, etc.
home_dir = os.path.expanduser("~")
primary_path = os.path.join(home_dir, "workspace", "aou-tutorial-notebooks", r_file_name)

try:
    # Attempt to open the resolved dynamic home directory path
    f = open(primary_path, "w")
    final_path = primary_path
except FileNotFoundError:
    # If the workspace directory structure doesn't exist, fallback to current working directory
    final_path = os.path.join(os.getcwd(), r_file_name)
    f = open(final_path, "w")


# --- WRITE CONTENT TO FILE ---
with f:
    f.write("# --- AUTO-GENERATED ENVIRONMENT SETUP ---\n\n")
    
    # Part 1: Write the Sys.setenv lines
    for var in vars_to_capture:
        value = os.environ.get(var)
        
        # Logic: If EXPORT_BUCKET is empty, use WORKSPACE_BUCKET value
        if var == "EXPORT_BUCKET" and not value:
            value = os.environ.get("WORKSPACE_BUCKET", "NOT_FOUND")
            
        if not value:
            value = "NOT_FOUND"
            
        f.write(f'Sys.setenv({var} = "{value}")\n')
    
    # Part 2: Verification Block
    f.write("\n# --- VERIFICATION BLOCK ---\n")
    
    r_list_str = ", ".join([f'"{v}"' for v in vars_to_capture])
    
    verification_code = f"""
vars_to_check <- c({r_list_str})

cat("\\n🔍 Current Workspace Variables:\\n")
for (v in vars_to_check) {{
  value <- Sys.getenv(v)
  # Prints the name (padded to 22 chars) and the value
  cat(sprintf("  %-22s : %s\\n", v, value))
}}
message("\\n✅ Environment Loaded Successfully.")
"""
    f.write(verification_code)

print(f"Done! {len(vars_to_capture)} variables saved to: {final_path}")


# # For SAS

# In[ ]:


import os

# Define the variables to capture from your Notebook VM
vars_to_capture = [
    "WORKSPACE_CDR",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_PROJECT",
    "WORKSPACE_BUCKET",
    "WORKSPACE_TEMP_BUCKET",
    "EXPORT_BUCKET",
    "bucket_aou_tutorial",     # Path (gs://...)
    "bucket_id_aou_tutorial",  # ID only (aou-tutorial-notebooks)
    "bucket_migrated",         # Path (gs://...)
    "bucket_id_migrated"       # ID only (rw-migration-xxxx)
]

sas_file_name = "Setting_Env_Variables.sas"

# --- DYNAMIC HOME-BASED PATH RESOLUTION ---
# Automatically resolves to /home/jupyter, /home/dataproc, etc.
home_dir = os.path.expanduser("~")
primary_sas_path = os.path.join(home_dir, "workspace", "aou-tutorial-notebooks", sas_file_name)

try:
    # Attempt to open the preferred path in the home workspace directory
    f = open(primary_sas_path, "w")
    final_sas_path = primary_sas_path
except FileNotFoundError:
    # Fallback to the current working directory if the above folder structure isn't there
    final_sas_path = os.path.join(os.getcwd(), sas_file_name)
    f = open(final_sas_path, "w")

# --- WRITE SAS CONTENT TO FILE ---
with f:
    f.write("/* --- AUTO-GENERATED ENVIRONMENT SETUP --- */\n\n")

    for var in vars_to_capture:
        value = os.environ.get(var)

        if var == "EXPORT_BUCKET" and not value:
            value = os.environ.get("WORKSPACE_BUCKET")

        if not value:
            value = "NOT_FOUND"

        value = value.replace(";", "")  # protect against accidental semicolons

        f.write(f"%let {var}={value};\n")

    f.write("\n/* --- VERIFICATION BLOCK --- */\n")

    for var in vars_to_capture:
        f.write(f"%put NOTE: {var}=&{var}.;\n")
        
print(f"Done! {len(vars_to_capture)} variables saved to: {final_sas_path}")

