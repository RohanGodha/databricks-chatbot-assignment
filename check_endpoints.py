from databricks.sdk import WorkspaceClient
import os

# Load environment variables
load_env_host = os.getenv("DATABRICKS_HOST")
load_env_token = os.getenv("DATABRICKS_TOKEN")

# Create Databricks client
w = WorkspaceClient(host=load_env_host, token=load_env_token)

# Get list of serving endpoints
eps = w.serving_endpoints.list()

# Display important details
for e in eps:
    name = e.name
    task = e.task
    status = e.state.ready.value  # e.g., 'READY'
    
    # Extract foundation model info from the config
    try:
        served_entity = e.config.served_entities[0]
        foundation_model = served_entity.foundation_model
        model_display_name = foundation_model.display_name if foundation_model else "N/A"
    except Exception:
        model_display_name = "N/A"

    print(f"{name} | {model_display_name} | Task: {task} | Status: {status}")



# from databricks.sdk import WorkspaceClient
# import os

# # Load environment variables
# load_env_host = os.getenv("DATABRICKS_HOST")
# load_env_token = os.getenv("DATABRICKS_TOKEN")

# # Create Databricks client
# w = WorkspaceClient(host=load_env_host, token=load_env_token)

# # Get list of serving endpoints (list of ServingEndpoint objects)
# eps = w.serving_endpoints.list()

# # Print names and full object as a fallback
# print([e.name + " -> " + str(e) for e in eps])

# # Loop through endpoints
# for e in eps:
#     name = e.name
#     # Depending on the structure, access other fields accordingly
#     # There is no `task` attribute in ServingEndpoint by default
#     print(name, "->", e)